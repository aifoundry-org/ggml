#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <random>

// This is a simple model with two tensors a and b
struct simple_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;
    struct ggml_tensor * n;

    // the context to define the tensor information (dimensions, size, memory data)
    struct ggml_context * ctx;
};

// Function to generate a Bernoulli noise matrix
void generate_bernoulli_noise(float* noise_matrix, int rows, int cols, float probability, float scale) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution bernoulli_dist(probability);

    for (int i = 0; i < rows * cols; i++) {
        noise_matrix[i] = bernoulli_dist(gen) * scale;
    }
}

// initialize the tensors of the model in this case three matrices 4x2, 2x3, 4x3
void load_model(simple_model & model, float * a, float * b, float * n, int rows_A, int cols_A, int rows_B, int cols_B) {
    size_t ctx_size = 0;
    {
        ctx_size += rows_A * cols_A * ggml_type_size(GGML_TYPE_F32); // tensor a
        ctx_size += rows_B * cols_B * ggml_type_size(GGML_TYPE_F32); // tensor b
        ctx_size += rows_B * cols_A * ggml_type_size(GGML_TYPE_F32); // tensor n
        ctx_size += 3 * ggml_tensor_overhead(), // tensors
        ctx_size += ggml_graph_overhead(); // compute graph
        ctx_size += 1024; // some overhead
        ctx_size += ctx_size;
    }

    struct ggml_init_params params {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false, // NOTE: this should be false when using the legacy API
    };

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_A, rows_A);
    model.b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_B, rows_B);
    model.n = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_A, rows_B);

    memcpy(model.a->data, a, ggml_nbytes(model.a));
    memcpy(model.b->data, b, ggml_nbytes(model.b));
    memcpy(model.n->data, n, ggml_nbytes(model.n));
}

// build the compute graph to perform a matrix multiplication and noise addition
struct ggml_cgraph * build_graph_noise_add(const simple_model& model) {
    struct ggml_cgraph  * gf = ggml_new_graph(model.ctx);

    // result = a*b+n^T
    struct ggml_tensor * result = ggml_add(model.ctx, ggml_mul_mat(model.ctx, model.a, model.b), model.n);

    ggml_build_forward_expand(gf, result);
    return gf;
}

// build the compute graph to perform a noisy matrix multiplication
struct ggml_cgraph * build_graph_mul_noise(const simple_model& model) {
    struct ggml_cgraph  * gf = ggml_new_graph(model.ctx);

    // result = a*b^T
    // float noise_scale = 10.0f;
    struct ggml_tensor * result = ggml_mul_mat_noisy(model.ctx, model.a, model.b);

    ggml_build_forward_expand(gf, result);
    return gf;
}


// compute with backend for noise multiplication
struct ggml_tensor * compute_mul_noise(const simple_model & model) {
    struct ggml_cgraph * gf = build_graph_noise_add(model);

    int n_threads = 1; // number of threads to perform some operations with multi-threading

    ggml_graph_compute_with_ctx(model.ctx, gf, n_threads);

    // in this case, the output tensor is the last one in the graph
    return ggml_graph_node(gf, -1);
}


// compute with backend for noise addition
struct ggml_tensor * compute_noise_add(const simple_model & model) {
    struct ggml_cgraph * gf = build_graph_noise_add(model);

    int n_threads = 1; // number of threads to perform some operations with multi-threading

    ggml_graph_compute_with_ctx(model.ctx, gf, n_threads);

    // in this case, the output tensor is the last one in the graph
    return ggml_graph_node(gf, -1);
}

int main(void) {
    ggml_time_init();

    // initialize data of matrices to perform matrix multiplication
    const int rows_A = 4, cols_A = 2;

    float matrix_A[rows_A * cols_A] = {
        2, 8,
        5, 1,
        4, 2,
        8, 6
    };

    const int rows_B = 3, cols_B = 2;
    /* Transpose([
        10, 9, 5,
        5, 9, 4
    ]) 2 rows, 3 cols */
    float matrix_B[rows_B * cols_B] = {
        10, 5,
        9, 9,
        5, 4
    };

    // Generate Bernoulli noise matrix for the required size (rows_B * cols_A)
    float matrix_N[rows_B * cols_A];
    float probability = 0.5; 
    float scale = 1.0;       

    generate_bernoulli_noise(matrix_N, rows_B, cols_A, probability, scale);

    simple_model model;
    load_model(model, matrix_A, matrix_B, matrix_N, rows_A, cols_A, rows_B, cols_B);

    // perform computation in cpu
    struct ggml_tensor * result_noise_add = compute_noise_add(model);
    struct ggml_tensor * result_mul_noise = compute_mul_noise(model);

    // get the result data pointer as a float array to print
    std::vector<float> out_data_add(ggml_nelements(result_noise_add));
    memcpy(out_data_add.data(), result_noise_add->data, ggml_nbytes(result_noise_add));


    std::vector<float> out_data_mul(ggml_nelements(result_mul_noise));
    memcpy(out_data_mul.data(), result_mul_noise->data, ggml_nbytes(result_mul_noise));

    // expected result:
    // [ 60.00 55.00 50.00 110.00
    //   90.00 54.00 54.00 126.00
    //   42.00 29.00 28.00 64.00 ]

    printf("mul mat with added noise(%d x %d) (transposed result):\n[", (int) result_noise_add->ne[0], (int) result_noise_add->ne[1]);
    for (int j = 0; j < result_noise_add->ne[1] /* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < result_noise_add->ne[0] /* cols */; i++) {
            printf(" %.2f", out_data_add[j * result_noise_add->ne[0] + i]);
        }
    }
    printf(" ]\n");

    printf("mul mat with injected noise(%d x %d) (transposed result):\n[", (int) result_mul_noise->ne[0], (int) result_mul_noise->ne[1]);
    for (int j = 0; j < result_mul_noise->ne[1] /* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < result_mul_noise->ne[0] /* cols */; i++) {
            printf(" %.2f", out_data_mul[j * result_mul_noise->ne[0] + i]);
        }
    }
    printf(" ]\n");


    // free memory
    ggml_free(model.ctx);
    return 0;
}

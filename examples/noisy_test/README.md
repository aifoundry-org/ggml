## Noisy context

This example simply performs a matrix multiplication, solely for the purpose of demonstrating a basic usage of ggml and backend handling. The code is commented to help understand what each part does.

As an addition to simple multiplication, some noise is added to the result of the multiplication to imitate noisy channel as partial implementation of Shannon's second theorem conditions.

Two implementations are provided:

1. Implemented new operation [`ggml_mul_mat_noisy`](../../src/ggml.c?plain=1#L5637) and apropriate forward method [`ggml_compute_forward_mul_mat_noisy`](../../src/ggml.c?plain=1#L12456).  However this implementation is not usefull, because I couldn't implement noise [scale](../../src/ggml.c?plain=1#L12655) modification outside of this forward method. May be because of the static graph execution paradigm.
2. Second approach is to directly add noise matrix to the result of multiplication directly adding aother operation into graph execution ([line](noisy-ctx.cpp?plain=1#L67)). The drawback is highly increased memory consumption, assuming that additional noise matrix should be stored somewhere. Although noise scale can be easely manipulated, it's still not clear how to pass it into loss function for further adjustements.


In that case operaton of matrix multiplication looks like this:

$$
A \times B + N = C + N
$$

$$
N \sim \text{Bernoulli}(p)
$$

$$
\begin{bmatrix}
2 & 8 \\
5 & 1 \\
4 & 2 \\
8 & 6 \\
\end{bmatrix}
\times
\begin{bmatrix}
10 & 9 & 5 \\
5 & 9 & 4 \\
\end{bmatrix}
+
N
=
\begin{bmatrix}
60 & 90 & 42 \\
55 & 54 & 29 \\
50 &  54 & 28 \\
110 & 126 & 64 \\
\end{bmatrix}
+ N
$$



The `simple-ctx` doesn't support gpu acceleration. 
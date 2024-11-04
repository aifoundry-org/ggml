#
# noisy-ctx

set(TEST_TARGET noisy-ctx)
add_executable(${TEST_TARGET} noisy-ctx.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)
#include <cuda_runtime.h>

__global__ void LowerTriangularMatMul(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    if (col <= row) {
        float sum = 0.0f;
        // Only k between col and row matters
        for (int k = col; k <= row; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    } else {
        C[row * N + col] = 0.0f;
    }
}

// input_a, input_b, output_c are device pointers
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t n) {    
    dim3 block(32, 32);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    LowerTriangularMatMul<<<grid, block>>>(input_a, input_b, output_c, n);
}

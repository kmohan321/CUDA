#include <cuda_runtime.h>

__global__ void LowerTriangularMatMul(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    if (col <= row) {
        float sum = 0.0f;
        for (int k = 0; k <= row; ++k) {
            if (k <= col) { 
                sum += A[row * N + k] * B[k * N + col];
            }
        }
        C[row * N + col] = sum;
    } else {
        C[row * N + col] = 0.0f;
    }
}

// Note: input_a, input_b, output_c are all device pointers to float32 arrays
extern "C" void solution(const float* input_a, const float* input_b, float* output_c, size_t n) {    

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    LowerTriangularMatMul<<<grid, block>>>(input_a, input_b, output_c, n);

}
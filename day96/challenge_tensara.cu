#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void matMulUpperTriangle(
    const float* __restrict__ matA,
    const float* __restrict__ matB,
    float* __restrict__ matC,
    int dimSize
) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if (row >= dimSize || col >= dimSize || row > col)
        return;

    float acc = 0.0f;

    for (int idx = row; idx <= col; ++idx) {
        acc += matA[row * dimSize + idx] * matB[idx * dimSize + col];
    }

    matC[row * dimSize + col] = acc;
}


extern "C" void solution(const float* matrixA,
              const float* matrixB,
              float* matrixC,
              size_t size)
{
    if (size == 0) return;

    dim3 tile(16, 16);
    dim3 grid((size + tile.x - 1) / tile.x,
              (size + tile.y - 1) / tile.y);

    matMulUpperTriangle<<<grid, tile>>>(matrixA, matrixB, matrixC, static_cast<int>(size));
    cudaDeviceSynchronize();
}

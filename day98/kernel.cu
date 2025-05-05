#include <cuda_runtime.h>

#define BLOCK_Y 32
#define BLOCK_X 32
#define BLOCK_K 16
#define WARP_X 16
#define WARP_Y 16
#define REG_Y 2
#define REG_X 2

__global__ void matmulSymmetric(
    const float* __restrict__ matA,    // N x N, symmetric
    const float* __restrict__ matB,    // N x N, symmetric
    float* __restrict__ matC,          // N x N
    int dim
) {
    __shared__ float tileA[BLOCK_Y][BLOCK_K];
    __shared__ float tileB[BLOCK_K][BLOCK_X];

    int threadX = threadIdx.x;
    int threadY = threadIdx.y;
    int globalRowBase = blockIdx.y * BLOCK_Y;
    int globalColBase = blockIdx.x * BLOCK_X;
    int regRowOffset = 2 * threadY;
    int regColOffset = 2 * threadX;

    float accum[REG_Y][REG_X] = {0.0f};

    for (int tileIdx = 0; tileIdx < dim; tileIdx += BLOCK_K) {
        int threadFlatIdx = threadY * WARP_X + threadX;
        for (int unroll = 0; unroll < 2; unroll++) {
            int globalIdx = threadFlatIdx + unroll * (WARP_X * WARP_Y);

            int loadRowA = globalIdx / BLOCK_K;
            int loadColA = globalIdx % BLOCK_K;
            int globalRowA = globalRowBase + loadRowA;
            int globalColA = tileIdx + loadColA;
            tileA[loadRowA][loadColA] = (globalRowA < dim && globalColA < dim) ? matA[globalRowA * dim + globalColA] : 0.0f;

            int loadRowB = globalIdx / BLOCK_X;
            int loadColB = globalIdx % BLOCK_X;
            int globalRowB = tileIdx + loadRowB;
            int globalColB = globalColBase + loadColB;
            tileB[loadRowB][loadColB] = (globalRowB < dim && globalColB < dim) ? matB[globalRowB * dim + globalColB] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_K; k++) {
            float aFrag[REG_Y];
            aFrag[0] = tileA[regRowOffset + 0][k];
            aFrag[1] = tileA[regRowOffset + 1][k];
            float bFrag[REG_X];
            bFrag[0] = tileB[k][regColOffset + 0];
            bFrag[1] = tileB[k][regColOffset + 1];

            accum[0][0] += aFrag[0] * bFrag[0];
            accum[0][1] += aFrag[0] * bFrag[1];
            accum[1][0] += aFrag[1] * bFrag[0];
            accum[1][1] += aFrag[1] * bFrag[1];
        }

        __syncthreads();
    }

    #pragma unroll
    for (int yi = 0; yi < REG_Y; yi++) {
        int rowOut = globalRowBase + regRowOffset + yi;
        if (rowOut >= dim) continue;
        #pragma unroll
        for (int xi = 0; xi < REG_X; xi++) {
            int colOut = globalColBase + regColOffset + xi;
            if (colOut >= dim) continue;
            matC[rowOut * dim + colOut] = accum[yi][xi];
        }
    }
}

extern "C" void runSymMatmul(
    const float* matA,
    const float* matB,
    float* matC,
    size_t dim
) {
    if (dim == 0) return;

    dim3 threads(WARP_X, WARP_Y);
    dim3 blocks((dim + BLOCK_X - 1) / BLOCK_X, (dim + BLOCK_Y - 1) / BLOCK_Y);

    matmulSymmetric<<<blocks, threads>>>(matA, matB, matC, (int)dim);
}

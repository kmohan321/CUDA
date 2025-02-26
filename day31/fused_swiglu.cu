#include <iostream>
#include <cuda.h>
#include <cmath>

#define TILE_SIZE 16

__global__ void fused_swiglu(float *m1, float *m2, float *output, 
                             int tile, int r, int c, int common) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    int x = threadIdx.x;
    int y = threadIdx.y;

    extern __shared__ float smem[];
    float *s1 = smem;
    float *s2 = s1 + tile * tile;
    float sum = 0.0f;

    for(int j = 0; j < (common + tile - 1) / tile; j++) {
        if(idy < r && (x + j * tile) < common)
            s1[x + y * tile] = m1[idy * common + x + j * tile];
        else
            s1[x + y * tile] = 0.0f;
        __syncthreads();

        if((y + tile * j) < common && idx < c)
            s2[x + y * tile] = m2[idx + y * c + c * tile * j];
        else
            s2[x + y * tile] = 0.0f;
        __syncthreads();

        if(idx < c && idy < r)
            for(int i = 0; i < tile; i++)
                sum += s1[i + y * tile] * s2[x + i * tile];
        __syncthreads();
    }
    if(idx < c && idy < r) {
        float swish_val = sum / (1.0f + expf(-sum)); //simply doing x*sigmoid(x)
        output[idx + idy * c] = swish_val * sum;
    }
}

int main() {
    int r = 32, c = 32, common = 32;
    size_t sizeA = r * common * sizeof(float);
    size_t sizeB = common * c * sizeof(float);
    size_t sizeC = r * c * sizeof(float);

    float *h_A = new float[r * common];
    float *h_B = new float[common * c];
    float *h_C = new float[r * c];

    for(int i = 0; i < r * common; i++) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for(int i = 0; i < common * c; i++) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((c + TILE_SIZE - 1) / TILE_SIZE, (r + TILE_SIZE - 1) / TILE_SIZE);
    size_t sharedMemSize = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);

    fused_swiglu<<<gridDim, blockDim, sharedMemSize>>>(d_A, d_B, d_C, TILE_SIZE, r, c, common);
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 5; i++) std::cout << h_C[i] << " ";
    std::cout << "\n";

    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}

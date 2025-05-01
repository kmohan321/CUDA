#include <cuda_runtime.h>
#include <math.h> 

__global__ void weighted_l2_norm_kernel(const float* __restrict__ x,
                              const float* __restrict__ w,
                              float* partial_sums,
                              int N) {
    extern __shared__ float shmem[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float sum = 0.0f;
    if (idx < N) {
        float val = x[idx];
        float weight = w[idx];
        sum = weight * val * val;
    }

    shmem[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset)
            shmem[tid] += shmem[tid + offset];
        __syncthreads();
    }

    if (tid == 0)
        partial_sums[blockIdx.x] = shmem[0];
}

extern "C" float compute_weighted_l2_norm(const float* d_x, const float* d_w, int N) {
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* d_partial;
    cudaMalloc(&d_partial, sizeof(float) * blocks);

    weighted_l2_norm_kernel<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
        d_x, d_w, d_partial, N);

    float* h_partial = (float*)malloc(sizeof(float) * blocks);
    cudaMemcpy(h_partial, d_partial, sizeof(float) * blocks, cudaMemcpyDeviceToHost);

    float total = 0.0f;
    for (int i = 0; i < blocks; ++i)
        total += h_partial[i];

    cudaFree(d_partial);
    free(h_partial);

    return sqrtf(total);
}

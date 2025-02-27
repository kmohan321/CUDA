#include <cuda.h>
#include <stdio.h>
#include <math.h>

#define THREADS_PER_BLOCK 256  
#define ELEMENTS_PER_THREAD 4  

__device__ float gelu(float x) {
    return 0.5 * x * (1.0 + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void layernorm_gelu(float* __restrict__ input, 
                               float* __restrict__ output, 
                               int C, float epsilon) {
    int row = blockIdx.x;
    __shared__ float mean_shared;
    __shared__ float var_shared;
    
    int start_idx = row * C;
    
    int tid = threadIdx.x;
    int step = blockDim.x * ELEMENTS_PER_THREAD;
  
    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int i = tid * ELEMENTS_PER_THREAD; i < C; i += step) {
        float val = input[start_idx + i];
        sum += val;
        sum_sq += val * val;
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }
    if (tid == 0) {
        mean_shared = sum / C;
        var_shared = rsqrtf(sum_sq / C - mean_shared * mean_shared + epsilon);
    }
    __syncthreads();
    
    for (int i = tid * ELEMENTS_PER_THREAD; i < C; i += step) {
        float norm = (input[start_idx + i] - mean_shared) * var_shared;
        output[start_idx + i] = gelu(norm);
    }
}


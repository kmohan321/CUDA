#include <cuda.h>
#include <stdio.h>
#include <math.h>

#define THREADS_PER_BLOCK 256 

__device__ float gelu(float x) {
    return 0.5 * x * (1.0 + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

__global__ void layernorm_gelu(float* __restrict__ input, 
                               float* __restrict__ output, 
                               int C, float epsilon) {
    int row = blockIdx.x;
    __shared__ float mean_shared[THREADS_PER_BLOCK];
    __shared__ float var_shared[THREADS_PER_BLOCK];
    
    int start_idx = row * C;
    int tid = threadIdx.x;
  
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = tid ; i < C; i += blockDim.x) {
        float val = input[start_idx + i];
        sum += val;
        sum_sq += val * val;
    }
    mean_shared[tid] = sum;
    var_shared[tid] = sum_sq;
    __syncthreads();

    // for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    //     sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    //     sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    // }
    for(int i = blockDim.x/2 ; i > 0 ; i/=2){
      if(tid<i){
        mean_shared[tid] += mean_shared[tid + i];
        var_shared[tid] += var_shared[tid + i];
      }
      __syncthreads();
    }

    float row_mean = mean_shared[0] / C;
    float row_var = rsqrtf(var_shared[0]/C - row_mean * row_mean + epsilon);

    for (int i = tid ; i < C; i += blockDim.x) {
        float norm = (input[start_idx + i] - row_mean) * row_var;
        output[start_idx + i] = gelu(norm);
    }
}


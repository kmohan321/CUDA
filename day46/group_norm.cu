#include<stdio.h>
#include<cuda.h>

template <unsigned int blockSize>
__device__ float blockReduceSum(float val) {
    __shared__ float shared[32];  // Shared memory for warp reduction
    int lane = threadIdx.x % warpSize;  
    int wid = threadIdx.x / warpSize;  

    // Perform warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);

    // Store per-warp results into shared memory
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    // Reduce within first warp
    if (wid == 0) {
        val = (lane < (blockSize / warpSize)) ? shared[lane] : 0;
        for (int offset = warpSize / 2; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }

    return val;
}

__global__ void group_norm(float *X, float *Y, float *gamma, float *beta, 
                           int B, int C, int HxW, int G, float eps,int blocksize) {
    int batch = blockIdx.x;  // Each block processes a batch
    int channel = blockIdx.y;  // Each block processes a channel
    int idx = threadIdx.x;  // Threads operate on spatial elements

    int group_id = channel / (C / G);  // Find group ID
    int group_offset = group_id * (C / G) * HxW;  // Offset for group

    __shared__ float mean, var;

    // Load values for this group
    float val = X[batch * C * HxW + channel * HxW + idx];

    // Step 1: Compute group mean
    float sum = blockReduceSum<blocksize>(val);
    if (idx == 0) mean = sum / (HxW * C / G);
    __syncthreads();

    // Step 2: Compute variance
    float diff = val - mean;
    float sq_sum = blockReduceSum<blocksize>(diff * diff);
    if (idx == 0) var = sqrtf(sq_sum / (HxW * C / G) + eps);
    __syncthreads();

    // Step 3: Normalize
    val = (val - mean) / var;
    if (gamma && beta) val = gamma[channel] * val + beta[channel];

    // Step 4: Store result
    Y[batch * C * HxW + channel * HxW + idx] = val;
}

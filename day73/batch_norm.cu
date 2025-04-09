#include <cuda_runtime.h>


//simply run parallely over the feature dimension 
__global__ void batch_normalised(float *input, float *output, int M, int N){

    int feature_idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if(feature_idx >= N) return;

    int batch_idx = threadIdx.y  //these represents the batch dimension
    extern __shared__ float smem[];
    float *mean_smem = smem;
    float *sq_mean = smem + blockDim.y;

    float local_sum = 0.0f;
    float local_variance = 0.0f;
    for(int i = 0; i < B; i+= blockDim.y){
        float curr_value = input[i * N + feature_idx];
        local_sum += curr_value;
        local_variance += curr_value * curr_value;
    }
    mean_smem[batch_idx] = local_sum;
    sq_mean[batch_idx] = local_variance;
    __syncthreads();

    









}





// Note: X, Y are all device pointers to float32 arrays
extern "C" void solution(const float* X, float* Y, size_t B, size_t F, size_t D1, size_t D2) {    
}
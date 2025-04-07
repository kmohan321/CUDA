#include <cuda_runtime.h>

__global__ void kl_div(float *predictions, float *targets, float *output, float N){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if(idx < N){
        float curr_prediction = predictions[idx];
        float curr_targets = targets[idx];
        output[idx] = curr_targets * (logf(curr_targets + 1e-10) - logf(curr_prediction + 1e-10));
    }
}



// Note: predictions, targets, output are all device pointers to float32 arrays
extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n) {    
}
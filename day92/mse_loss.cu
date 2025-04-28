#include <cuda_runtime.h>

__global__ void MSE(const float * __restrict__ predictions, float *__restrict__ targets , float *__restrict__ output, int N, int M){

    int row = blockIdx.x;
    int idx = threadIdx.x;

    extern __shared__ float smem[];

    float local_sum = 0.0f;
    for(int i = idx; i<N; i += blockDim.x){
        float curr_pred = predictions[row * N + i];
        float curr_target = targets[row * N + i];
        float diff = curr_pred - curr_target;
        local_sum += diff * diff;
    }
    smem[idx] = local_sum;

    //reduction 
    for(int stride = blockDim.x/2; stride>0; stride /=2){
        if(idx <stride ){
            smem[idx] += smem[idx + stride];
        }
        __syncthreads();
    }
    if(idx==0){
        output[row] = smem[0];
    }

}

__global__ void BlockSum(float *input, float *output, int N){

    int idx = threadIdx.x;
    

    float block_local_sum = 0.0f;
    for(int i = 0; )
    

}


// Note: predictions, targets, output, shape are all device pointers to float32 arrays
extern "C" void solution(const float* predictions, const float* targets, float* output, size_t* shape, size_t ndim) {    
}
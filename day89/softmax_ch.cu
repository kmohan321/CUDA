#include <cuda_runtime.h>

__global__ void softmax(const float __restrict__ *input, float __restrict__ *output, int M, int N, int dim, float *shape, float *ndim){

    int row = blockIdx.x; //will give the elements over the remaining dimension, need reformulation
    int idx = threadIdx.x; 

    



    extern __shared__ float smem[];
    for(int stride = idx; stride < N; stride+=blockDim.x){

    }
}





// Note: input, output, shape are all device pointers to float32 arrays
extern "C" void solution(const float* input, int dim, float* output, size_t* shape, size_t ndim) {    
}
#include <cuda_runtime.h>

__global__ void tanh(const float *__restrict__ input, float * __restrict__ output, int n, int m){

    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if(col>=n/4 && row>=m) return;
    int offset = row * n/4 + col;

    //loading the multiple elements 
    float4 curr_val = *(reinterpret_cast<const float4*>(input) + offset);

    float4 out;
    out.x = __tanhf(curr_val.x);
    out.y = __tanhf(curr_val.y);
    out.z = __tanhf(curr_val.z);
    out.w = __tanhf(curr_val.w);

    *(reinterpret_cast<float4*>(output) + offset) = out;

}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m) { 
    dim3 blocksize(8,32);
    dim3 grid((n/4 + blocksize.x  -1 )/blocksize.x, (m  + blocksize.y -1)/blocksize.y);
    tanh<<<grid,blocksize>>>(input, output, n,m);   
}
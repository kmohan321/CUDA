#include <cuda_runtime.h>

//let's try loading multiple elements
__global__ void sigmoid(const float * __restrict__ input, float * __restrict__ output, int M, int N){
    
    int row = blockIdx.x;
    int idx = threadIdx.x;
    int row_offset = row * N;

    if(row >= M) return;
    for(int stride = idx*4; stride < N; stride += blockDim.x*4){
        float4 *curr_idx = (float4*)&input[row_offset + stride];
        float4 curr_val = *curr_idx;

        output[row_offset + stride] = 1.0f / (1.0f + __expf(-curr_val.x));
        output[row_offset + stride] = 1.0f / (1.0f + __expf(-curr_val.y));
        output[row_offset + stride] = 1.0f / (1.0f + __expf(-curr_val.z));
        output[row_offset + stride] = 1.0f / (1.0f + __expf(-curr_val.w));
    }
}


// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {    

    int threads = 512;
    dim3 blocksize(threads);
    dim3 grid (m);
    sigmoid<<<grid, blocksize>>>(input,output,m,n);

}

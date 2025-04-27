#include <cuda_runtime.h>
#define PIE 3.14159

__global__ void GELU(const float *__restrict__ input, float *__restrict__ output, int N, int M){
    
    int row = blockIdx.x;
    int idx = threadIdx.x;
    int row_offset = row*N;
    for(int stride = idx; stride < N; stride += blockDim.x){
        float curr_value = input[row_offset + stride];
        float value_3 = curr_value * curr_value * curr_value;
        float tanh_value = tanhf(sqrtf(2/PIE) * (curr_value + 0.044715*value_3));
        output[row_offset + stride] = 0.5 * curr_value * (1.0f + tanh_value);
    }

}
// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {  

    int threads = 512;
    dim3 blocksize(threads);
    dim3 grid(m);
    GELU<<<grid, blocksize>>>(input, output, n,m);  
}
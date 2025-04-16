#include <cuda_runtime.h>

__global__ void ELU(const float *input, float *output, int M, int N, float alpha){
    
    int idx  = threadIdx.x;
    int row = blockIdx.x;
    int row_offset = row * N;

    if(row >= M) return;
    for(int i = idx; i < N; i += blockDim.x){
        float curr_value = input[row_offset + i];
        if(curr_value > 0){
            output[row_offset + i] = curr_value;
        }
        else{
            output[row_offset + i] = alpha * (__expf(curr_value) - 1.0f);
        }
    }
}


// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m, float alpha) {    

    int threads = 512;
    dim3 blocksize(threads);
    dim3 grid(m);
    ELU<<<grid, blocksize>>>(input,output,m,n,alpha);

}
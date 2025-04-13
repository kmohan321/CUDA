#include <cuda_runtime.h>

__global__ void rms(const float *input, float *output, int B, int N){

    int bidx = blockIdx.x;
    int idx = threadIdx.x;

    extern __shared__ float smem[];

    if(bidx >= B) return;
    float local_sum = 0.0f;
    for(int i = idx ; i < N; i+=blockDim.x ){
        float curr_value = input[bidx * N + i];
        local_sum += curr_value * curr_value;
    }

    smem[idx] = local_sum;
    __syncthreads();
    for(int stride = blockDim.x/2 ; stride>0 ; stride /=2){
        if(idx < stride){
            smem[idx] += smem[idx + stride];
        }
        __syncthreads();
    }

    float variance  = (smem[0]/N);
    float r_variance = sqrtf(variance + 1e-5);
    for(int i = idx ; i < N; i += blockDim.x ){
        float curr_value = input[bidx * N + i];
        output[bidx * N + i] = curr_value / r_variance;
        
    }

}


// Note: X, Y are all device pointers to float32 arrays
extern "C" void solution(const float* X, float* Y, size_t B, size_t N) {    

    int threads = 1024;
    dim3 blocksize(threads);
    dim3 grid(B);
    int smem_size = threads * sizeof(float);
    rms<<<grid, blocksize, smem_size>>>(X,Y,B,N);

}
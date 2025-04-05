#include <cuda_runtime.h>

__global__ void cumulative_sum(const float *input, float *out, float *block_sum, int N){
    
    int x = threadIdx.x;
    int b_idx = blockIdx.x;
    int idx = x + b_idx * blockDim.x;
    extern __shared__ float smem[];

    if(idx < N){
        smem[x] = input[idx];
    }
    __syncthreads();

    for(int stride = 1; stride < blockDim.x/2; stride *=2){
        if(x>stride){
            smem[x] += smem[x-stride];
        }
        __syncthreads();
    }

    if(idx<N){
        out[idx] = smem[x];
    }

    if(x==0 && idx < N){
        block_sum[b_idx] = smem[N-(b_idx * blockDim.x)];
    }

}

__global__ void block_level(float *out, float *block_sum, int N){

        int x = threadIdx.x;
        int b_idx = blockIdx.x;
        int idx = x + b_idx * blockDim.x;

        float correction_factor = 0.0f;
        if(b_idx != 0){
            correction_factor += block_sum[b_idx-1];
            out[idx] += correction_factor;
        }

}


// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t N) {    

    int threads = 512;
    dim3 grid((N + threads -1)/threads);
    dim3 blocksize(threads);
    float *d_block_sum;
    cudaMalloc((void**)&d_block_sum , grid.x * sizeof(float));
    int smem_size = threads * sizeof(float);
    cumulative_sum<<<grid,blocksize,smem_size>>>(input, output, d_block_sum,N);
    block_level<<<grid,blocksize>>>(output, d_block_sum,N);


}
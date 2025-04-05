#include <cuda_runtime.h>

__global__ void cumulative_sum(const float *input, float *out, float *block_sum, int N){
    
    int x = threadIdx.x;
    int b_idx = blockIdx.x;
    int idx = x + b_idx * blockDim.x;
    extern __shared__ float smem[];

    if(idx < N){
        smem[x] = input[idx];
    }
    else{
        smem[x] = 0.0f;
    }
    __syncthreads();

    for(int stride = 1; stride < blockDim.x; stride *=2){
        float temp = 0.0f;  
        if(x >= stride){
            temp = smem[x] + smem[x-stride]; 
          }
          __syncthreads();
          
          if(x >= stride){
            smem[x] = temp;
          }
          __syncthreads();
  
      }

    if(idx<N){
        out[idx] = smem[x];
    }

    if(x==blockDim.x-1 && idx < N){
        block_sum[b_idx] = smem[x];
    }

}

__global__ void block_level(float *out, float *block_sum, int N){

    int x = threadIdx.x;
    int b_idx = blockIdx.x;
    int idx = x + b_idx * blockDim.x;

    if(b_idx != 0 && idx < N){
        float correction_factor = block_sum[b_idx-1];
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
    float *block_sum = (float*)malloc(grid.x * sizeof(float));
    cudaMemcpy(block_sum,d_block_sum,grid.x * sizeof(float), cudaMemcpyDeviceToHost);
    for(int i= 1; i < grid.x; i++){
        block_sum[i] = block_sum[i-1] + block_sum[i];
    }
    cudaMemcpy(d_block_sum,block_sum,grid.x * sizeof(float), cudaMemcpyHostToDevice);
    block_level<<<grid,blocksize>>>(output, d_block_sum,N);


}
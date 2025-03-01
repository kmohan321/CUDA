#include "stdio.h"
#include <cuda_runtime.h>

__global__ void reduction_kernel (float *input, float *output,int N){

    extern  __shared__ float smem[];
    int idx  = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N){
        smem[threadIdx.x] = input[idx];
    }
    else{
        smem[threadIdx.x] = 0.0f;
    }
    __syncthreads();
    
    for(int i = blockDim.x/2; i>0; i/=2){
        if(threadIdx.x<i){
            smem[threadIdx.x]+= smem[threadIdx.x + i];
        }
        __syncthreads();
    }
    if(threadIdx.x==0){
        output[blockIdx.x] = smem[0];
    }
}


void solve(const float* input, float* output, int N) {  

    dim3 blocksize(1024);
    int blocks = (N + 1024 -1)/1024;
    dim3 grid(blocks);

    float *input_dev , *output_dev;
    cudaMalloc((void**)&input_dev, N*sizeof(float));
    cudaMalloc((void**)&output_dev, blocks*sizeof(float));

    cudaMemcpy(input_dev,input,N*sizeof(float),cudaMemcpyHostToDevice);
    
    int smem_size = 1024*sizeof(float);
    reduction_kernel<<< grid,blocksize,smem_size>>>(input_dev,output_dev,N);
    cudaDeviceSynchronize();

    float *out_h = (float*)malloc(blocks*sizeof(float));
    cudaMemcpy(out_h,output_dev,blocks*sizeof(float),cudaMemcpyDeviceToHost);

    //final operation on cpu
    float final_sum = 0.0f;
    for(int i = 0; i<blocks;i++){
        final_sum += out_h[i];
    }
    *output = final_sum;

    cudaFree(input_dev);
    cudaFree(output_dev);
    free(out_h);


}
#include<stdio.h>
#include<cuda.h>

#define smem_size 32
#define N 100

__global__ void prefix_sum(float *input, float *output,float *block_level){

    int x = threadIdx.x;
    int idx = x + blockDim.x * blockIdx.x;
    __shared__ float smem[smem_size];
    
    if(idx < N){
        smem[x] = input[idx]; 
    }
    else{
      smem[x] = 0.0f;
    }
    __syncthreads();

    for(int stride = 1; stride < smem_size; stride *=2){
        if(x >= stride){
          smem[x] = smem[x] + smem[x-stride];
        }
        __syncthreads();
    }
    if(idx < N){
      output[idx] = smem[x];
    }
    if(x==smem_size-1){
      block_level[blockIdx.x] = smem[x];
    }
}
__global__ void block_reduction(float *output, float*block_sum){
      int idx = threadIdx.x + blockIdx.x * blockDim.x;
      
      if(idx<N && blockIdx.x!=0){
        output[idx] += block_sum[blockIdx.x-1];
      }
}


void inclusive_scan(float *d_input, float *d_output) {
  int threadsPerBlock = smem_size;
  const int blocksPerGrid = (N + smem_size - 1) / smem_size;

  float *d_block_sums, *d_block_sums_scan;
  cudaMalloc(&d_block_sums, blocksPerGrid * sizeof(float));
  cudaMalloc(&d_block_sums_scan, blocksPerGrid * sizeof(float));

  prefix_sum<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_block_sums);

  float h_block_sums[blocksPerGrid];
  cudaMemcpy(h_block_sums, d_block_sums, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
  
  for (int i = 1; i < blocksPerGrid; i++) {
      // printf("%.2f",h_block_sums[i - 1]);
      h_block_sums[i]+= h_block_sums[i-1];
  }

  cudaMemcpy(d_block_sums_scan, h_block_sums, blocksPerGrid * sizeof(float), cudaMemcpyHostToDevice);

  block_reduction<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_block_sums_scan);

  cudaFree(d_block_sums);
  cudaFree(d_block_sums_scan);
}

int main() {
  float h_input[N], h_output[N]; 
  float *d_input, *d_output; 


  for (int i = 0; i < N; i++) {
      h_input[i] = 1.0f; 
  }

  cudaMalloc((void **)&d_input, N * sizeof(float));
  cudaMalloc((void **)&d_output, N * sizeof(float));

  cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

  inclusive_scan(d_input, d_output);

  cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

  printf("Prefix sum result:\n");
  for (int i = 0; i < N; i++) {
      printf("%.1f ", h_output[i]);
  }
  printf("\n");

  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
}
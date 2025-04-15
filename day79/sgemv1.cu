#include<stdio.h>
#include<cuda.h>

/*
- naive approach -> one thread will calculate the one element 
- N is the common dimension
*/
// template <int M, int N>
// __global__ void sgemv(float *matrix, float *vector, float *ouput){

//     int row = threadIdx.x + blockDim.x * blockIdx.x;
//     int offset = row * N;
//     if(row < M){
//       float local_sum = 0.0f;
//       for(int i = 0; i < N; i++){
//         local_sum += matrix[offset + i] * vector[i];
//       }
//       ouput[row] = local_sum;
//     }
// }
/*
approach 2 - let's use the shared memory for storing the things
- tiled approach we will laod a tile of vector and then will do the operation
*/
template <int M, int N, int tile>
__global__ void sgemv_2(float *matrix, float *vector, float *ouput){

    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int offset = row * N;
    extern __shared__ float smem[]; //size -> blockdim.x -> tile

    float local_sum = 0.0f;
    for(int tileid = 0; tileid < (N + tile - 1)/tile; tileid++){
      if(threadIdx.x < tile && (tileid * tile + threadIdx.x ) < N){
          smem[threadIdx.x] = vector[tileid * tile + threadIdx.x];
      }
      __syncthreads();
      
      if(row < M){
        for(int i = 0; i < tile; i++){
          if((tileid *tile + i))<N{
            local_sum += matrix[offset + tileid *tile + i] * smem[tileid * tile + i];
          }
        }
      }
      __syncthreads();
    } 

}

float compute_gflops(int M, int N, float ms) {
  return (2 * M * N) / (ms * 1e6);
}

float compute_peak_gflops(float gflops, float THEORETICAL_MAX_GFLOPS) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  return (gflops / THEORETICAL_MAX_GFLOPS) * 100;
}

float compute_peak_memory_bandwidth(int M, int N, float ms, float THEORETICAL_MAX_MEMORY_BANDWIDTH) {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  size_t totalFloats = (size_t)(M * N + N + M);
  float totalBytes = (float)totalFloats * sizeof(float);

  float secs = ms / 1000.0f;
  float gbPerSec = (totalBytes / secs) / 1.0e9;

  return (gbPerSec / THEORETICAL_MAX_MEMORY_BANDWIDTH) * 100;
}

void print_kernel_essentials(int M, int N, float ms, float THEORETICAL_MAX_GFLOPS, float THEORETICAL_MAX_MEMORY_BANDWIDTH) {
  float gflops = compute_gflops(M, N, ms);
  printf(">> Execution time: %f ms\n", ms);
  printf(">> Achieved (GFLOPS): %f\n", gflops);
  printf(">> Theoretical max (GFLOPS): %f\n", THEORETICAL_MAX_GFLOPS);
  printf(">> Maximum memory bandwidth: %f GB/s\n", THEORETICAL_MAX_MEMORY_BANDWIDTH);
  printf(">> Achieves %f %% of peak GFLOPS\n", compute_peak_gflops(gflops, THEORETICAL_MAX_GFLOPS));
  printf(">> Achieves %f %% of peak Memory Bandwidth\n", compute_peak_memory_bandwidth(M, N, ms, THEORETICAL_MAX_MEMORY_BANDWIDTH));
}


int main() {
  const int M = 4;
  const int N = 3;
  const int tile = 256;

  float h_matrix[M * N] = {
      1, 2, 3,
      4, 5, 6,
      7, 8, 9,
      10, 11, 12
  };
  float h_vector[N] = {1, 1, 1};
  float h_output[M] = {0};

  float *d_matrix, *d_vector, *d_output;
  cudaMalloc(&d_matrix, M * N * sizeof(float));
  cudaMalloc(&d_vector, N * sizeof(float));
  cudaMalloc(&d_output, M * sizeof(float));

  cudaMemcpy(d_matrix, h_matrix, M * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_vector, h_vector, N * sizeof(float), cudaMemcpyHostToDevice);

  dim3 blockDim(256);
  dim3 gridDim((M + blockDim.x - 1) / blockDim.x);
  int smem_size = blockDim.x * sizeof(float);
  
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  sgemv_2<(M, N, tile)><<<gridDim, blockDim, smem_size>>>(d_matrix, d_vector, d_output);
  cudaEventRecord(stop);

  cudaMemcpy(h_output, d_output, M * sizeof(float), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);

  printf("Result (Matrix * Vector):\n");
  for (int i = 0; i < M; ++i) {
      printf("%.f\n", h_output[i]);
  }

  // Theoretical values (adjust for your GPU)
  float THEORETICAL_MAX_GFLOPS = 10000.0f;              // Example: NVIDIA A100 ~19.5 TFLOPS (FP32)
  float THEORETICAL_MAX_MEMORY_BANDWIDTH = 1555.0f;     // GB/s (adjust as per GPU, ex: A100 HBM2)

  print_kernel_essentials(M, N, ms, THEORETICAL_MAX_GFLOPS, THEORETICAL_MAX_MEMORY_BANDWIDTH);

  cudaFree(d_matrix);
  cudaFree(d_vector);
  cudaFree(d_output);

  return 0;
}



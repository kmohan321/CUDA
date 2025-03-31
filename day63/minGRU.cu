#include<stdio.h>
#include<cuda.h>

__global__ void matmul(float *A, float *B, float *out, int M, int N,int K,int tile, bool is_sigmoid){

      int x = threadIdx.x;
      int y = threadIdx.y;
      int col = x + blockDim.x * blockIdx.x;
      int row = y + blockDim.y * blockIdx.y;
      
      extern __shared__ float smem[];

      float *tile_A = smem;
      float *tile_B = smem + tile*tile;

      //move over the tiles
      float sum = 0.0f;
      for(int tile_id = 0; tile_id<(K+tile-1)/tile; tile_id++){

        //loading the tiles in shared memory (using the threads to load parallely)
        if(row < M && (x + tile_id * tile) <K){
          tile_A[y * tile + x] = A[row * K + (x + tile_id * tile)];
        }
        else{
          tile_A[y*tile + x] = 0.0f;
        }

        if(col < N && (y + tile_id * tile)<K){
          tile_B[y * tile + x] = B[(y + tile_id*tile)*N + col]; //row *stride + col
        }
        else{
          tile_B[y * tile + x] = 0.0f;
        }
        __syncthreads();

        //computing the local sum
        for(int k = 0; k<tile; k++){
          sum += tile_A[y * tile + k] * tile_B[k * tile + x];
        }
        __syncthreads();
      }
      if(row < M && col <N){
        if(is_sigmoid){
          out[row *N + col] = 1.0f / (1.0f + expf(-sum));
        }
        else{
          out[row *N + col] = sum;
        }
      }
}

// Parallel Scan for Recurrence: h_t = (1 - z_t) h_{t-1} + z_t h_tilde
__global__ void parallel_scan(float *z, float *h_tilde, float *h_out, int batch_size, int seq_len, int hidden_size) {
  int batch = blockIdx.x;
  int hidden = threadIdx.x;

  if (hidden >= hidden_size) return;

  float h_prev = h_out[batch * hidden_size  + hidden]; // Initial h_0

  for (int t = 0; t < seq_len; t++) {
      int idx = batch * seq_len * hidden_size + t * hidden_size + hidden;
      float z_t = z[idx];
      float h_tilde_t = h_tilde[idx];

      h_prev = (1 - z_t) * h_prev + z_t * h_tilde_t;
      h_out[idx] = h_prev;  // Store the updated hidden state
  }
}





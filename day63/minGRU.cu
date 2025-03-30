#include<stdio.h>
#include<cuda.h>

__global__ void matmul(float *A, float *B, float out, int M, int N,int K,int tile){

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

        //loading the tiles in shared memory (we will use the threads to load parallely)
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
          sum += tile_A[row * tile + k] * tile_B[k * tile + col];
        }
        __syncthreads();
      }
}


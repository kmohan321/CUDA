#include<stdio.h>
#include<cuda.h>

//1d block tiling 
/*
- simple we want to load multiple elements for BM direction using less threads
- should launch kernel in such a way that that we load in shared memory easily
- we are laucnhing the blocksize as (BM/bm * BN) ??? this is wrong as you can see when you set BN threads as column 
in case of loading B and then k = remaining threads but in case of A loading where you set BK so theere is mismatch here
- correct way is to use launch config as BM * BK = BN * BK = blocksize
*/
template <int BM, int BN, int BK, int bm>
__global__ void tiled_matmul_1d(float *A, float *B, float *out, int M, int N, int K){

    int idx = threadIdx.x;
    
    int row_offset = blockIdx.x * blockDim.x;
    int col_offset = blockIdx.y * blockDim.y;

    extern __shared__ float smem[];
    float *A_tile = smem;
    float *B_tile = smem + BM *BK;

    int rowA = idx / BK; //A - > BM * BK  
    int colA = idx % BK; 

    int rowB = idx / BN; //B -> BK * BN
    int colB = idx % BN;

    int blockrow = idx / BN; //C -> BM*BN
    int blockcol = idx % BN;

    float local_sum[bm] = {0.0};
    for(int tile_id  = 0 ; (K + BK - 1)/BK; tile_id++){
      //laoding A into the shared memory

      if((row_offset + rowA) < M && (colA + tile_id * BK) < K){
        A_tile[rowA * BK + colA] = A[(row_offset + rowA) * K + (colA + tile_id * BK)];
      }
      else{
        A_tile[rowA * BK + colA] = 0.0f;
      }
      //loading B into the shared memory 
      if((rowB + tile_id * BK) < K && (colB + col_offset)< N){
        B_tile[rowB * BN + colB] = B[(rowB + tile_id * BK) * N + (colB + col_offset)];
      }
      else{
        B_tile[rowB * BN + colB] = 0.0f;
      }
      __syncthreads();

      for(int i = 0; i<bm; bm++){
        for(int j = 0; j<BK; j++){
          local_sum[bm] += A_tile[(blockrow*bm + i) * BK + j] * B_tile[j * BN + blockcol]; 
        }
      }
      __syncthreads();

    }

    for(int i = 0; i<bm; bm++){
      out[(row_offset + blockrow * bm + i) * N + blockcol] = local_sum[i];
    }
}
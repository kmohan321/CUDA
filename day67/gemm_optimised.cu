#include<stdio.h>
#include<cuda.h>

/*
this is naive implementation 
- Output matrix is distributed over the grid (M,N)
- It means each thread calculates the one element of the output (Cij)-> i th row of A 
and j the column of B 
- We are looping over the common dimension K
*/

/*
- Accessing the row again and again from the global memory of A for calculating the first row 
of the output  
*/
__global__ void naive_matmul(float *A, float *B, float *C, int M, int N, int K){

  int row = threadIdx.y + blockDim.y * blockIdx.y;
  int col = threadIdx.x + blockDim.x * blockIdx.x;

  if(row < M && col < N){
    float sum = 0.0f;
    for(int k = 0; k < K; k++){
      sum += A[row * K + k] * B[k*N + col];
    }
    C[row * N + col] = sum;
  }
}

/*
- instead of accessing global memory, we minimized the access by caching the tiles of A and B
in shared memory for faster access(low latency)
- considering tile area equal to block area here
- each thread is computing the just single element
*/ 

__global__ void smem_matmul(float *A, float *B, float *out, int M, int N,int K,int tile){

  int x = threadIdx.x;
  int y = threadIdx.y;
  int col = x + blockDim.x * blockIdx.x;
  int row = y + blockDim.y * blockIdx.y;
  
  extern __shared__ float smem[]; //for caching the data 

  float *tile_A = smem;
  float *tile_B = smem + tile*tile;

  //move over the tiles
  float sum = 0.0f;
  for(int tile_id = 0; tile_id < (K+tile-1)/tile; tile_id++){

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
    for(int k = 0; k < tile; k++){
      sum += tile_A[y * tile + k] * tile_B[k * tile + x];
    }
    __syncthreads();
  }

  if(row < M && col <N){
    out[row *N + col] = sum;
  }
}

/*
- same approach as above instead of computing one element per thread we are going
 to compute the multiple elements
- bm * bn is the size for the each thread will compute
- BM * BN is the tile size (no of elements of output tile)
- we have to load BM * BK tile of A and B, to do this we will use multiple loadings per thread
- required threads -> (BM*BN) / bm*bn thread configuration -> (BM/bm, BN/bn)
*/

__global__ void matmul(float *A, float *B, float *out, int M, int N, int K, int BM, int BN,int BK, int bm, const int bn){

    int idx = threadIdx.x;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;

    int row_start = bidy * BM;
    int col_start = bidx * BN;

    extern __shared__ float smem[]; //for caching the data

    float *tile_A = smem;
    float *tile_B = smem + BM*BK;

    int numthreads = (BM*BN) / (bm*bn);
    int strideA = numthreads / BK;
    int strideB = numthreads / BN;


    int rowA = idx / BK; //threads for A 
    int colA = idx % BK;

    int rowB = idx / BN; //threads for B
    int colB = idx % BN;

    float row_local[bm] = {0.0};
    float col_local[bn] = {0.0};

    float sum[bm*bn] = {0.0};

    for(int tile_id = 0; tile_id < (K + BK - 1)/BK; tile_id++){

        //loading A into the shared memory (BM*BK)
        for(int loadoffset = 0; loadoffset < BM; loadoffset += strideA){ //load multiple rows (along BM)
          if((row_start + rowA + loadoffset) < M && (colA + tile_id *BK) < K){
            tile_A[(rowA + loadoffset) * BK + colA] = A[(row_start + rowA + loadoffset)*K + (colA + tile_id *BK)];
          }
          else{
            tile_A[(rowA + loadoffset) * BK + colA] = 0.0f;
          }
        }
        //laoding B into shared memory (BK*BN)
        for(int loadoffset = 0; loadoffset < BK; loadoffset += strideB){ //load multiple rows (along BK)
          if((col_start + colB) < N && (rowB + tile_id *BK + loadoffset) < K){
            tile_B[(rowB + loadoffset) * BN + colB] = B[(rowB + loadoffset + tile_id * BK) * N + (col_start + colB)];
          }
          else{
            tile_B[(rowB + loadoffset) * BN + colB] = 0.0f;
          }
        }

        __syncthreads();
        //we laoded BM*BK and BK*BN
        //now the issue is that we have to load multiple values of row BM and col BN
        //let's try to loop over BK 
        for(int j =0 ; j < BK; j++){
          
          //let's load all the row for this thread in registers
          for(int i =0; i < bm;i++){
            row_local[i] = tile_A[(rowA + i) * BK + j ];
          }
          //let's load all thee col for this thread in registers
          for(int i = 0; i < bn; i++){
            col_local[i] = tile_B[j * BN + (colB + i)];
          }

          //looping to get local sum
          for(int i = 0; i<bm ; i++){
            for(int j = 0; j<bn; j++){
              sum[i * bn + j] += row_local[i] * col_local[j];
            }
          }
        }
    }

    for(int i = 0; i<bm ; i++){
      for(int j = 0; j<bn; j++){
        out[(row_start + rowA + i)*N + (col_start + colB + j)] = sum[i * bn + j];
      }
    }

}




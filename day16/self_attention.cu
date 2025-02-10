#include<stdio.h>
#include<cuda.h>

#define batch 1  // Batch size
#define s 4  // Sequence length
#define d 4  // Embedding dimension
#define BLOCKSIZE 2  // Block size for tiled computation
/*
- will take the input x shape (B,S,D)
- q = x @ W_Q
- k = x @ W_K
- v = x @ W_V
- these are the projections of Query,Key and Values
- shape -> (B,S,d)
*/

/* this first kernel will perform the matrix multplication
- tiled matrix multiplication
- input will be of shape (B,S,D)
- here B will be handle by each Z dim block 
- and S will be handle by each Y dim block
- and D will be handle by each X dim block
*/
__global__ void GMM (float *A ,float *B ,float *out, int blocksize, int M, int N, int K){

    int x = threadIdx.x;
    int y = threadIdx.y;

    int BATCH = blockIdx.z;
    int offset = BATCH*M*N;

    int idx = x + blockDim.x * blockIdx.x; // for columns 
    int idy = y + blockDim.y * blockIdx.y; // for rows

    extern __shared__ float sharedMem[];
    float *a = sharedMem;
    float *b = sharedMem + blocksize * blocksize;

    float sum = 0.0f;
    for (int tileid = 0; tileid < (K+blocksize-1)/blocksize; tileid++){

        if(idy < M && (x + blocksize*tileid)<N){
          a[x + y*blocksize] = A[offset + x + tileid*blocksize + idy * K];
        }
        else{
          a[x + y*blocksize] = 0.0f;
        }
        __syncthreads();
        
        if((y + tileid*blocksize)<K && idx <N){
          b[x + y*blocksize] = B[offset + idx + tileid*blocksize*N + y*N];
        }
        else{
          b[x + y*blocksize] = 0.0f;
        }
        __syncthreads();

        for(int i =0 ; i<blocksize; i++){
          sum += a[i + y*blocksize] * b[x + i*blocksize];
        }
        __syncthreads();
    }
    if(idy < M && idx < N){
      out[offset + idx + idy *N] = sum;
    }
}

void QKV_proj(float *x, float *Q, float *K, float*V, int B, int S, int D, int blocksize,
        float *Wq, float *Wk, float *Wv){

      int blockz  = B;
      int blocky = (S + blocksize - 1) / blocksize;
      int blockx = (D + blocksize - 1) / blocksize;
      dim3 gridsize(blockx,blocky,blockz);
      dim3 BlockSize(blocksize,blocksize);

      int shared_mem_size = 2 * blocksize * blocksize * sizeof(float);

      GMM<<<gridsize , BlockSize, shared_mem_size>>>(x,Wq,Q,blocksize,S,D,D);
      GMM<<<gridsize , BlockSize, shared_mem_size>>>(x,Wk,K,blocksize,S,D,D);
      GMM<<<gridsize , BlockSize, shared_mem_size>>>(x,Wv,V,blocksize,S,D,D);

}
//just for random values between 0 and 1
void randomInit(float *data, int size) {
  for (int i = 0; i < size; i++) {
      data[i] = (float)(rand() % 10) / 10.0f; 
  }
}

int main() {
  
  float *x, *Wq, *Wk, *Wv, *Q, *K, *V;
  int input_size = batch * s * d * sizeof(float);

  cudaMallocManaged(&x, input_size);
  cudaMallocManaged(&Wq, input_size);
  cudaMallocManaged(&Wk, input_size);
  cudaMallocManaged(&Wv, input_size);
  cudaMallocManaged(&Q, input_size);
  cudaMallocManaged(&K, input_size);
  cudaMallocManaged(&V, input_size);

  randomInit(x, batch * s * d);
  randomInit(Wq, d * d);
  randomInit(Wk, d * d);
  randomInit(Wv, d * d);

  QKV_proj(x, Q, K, V, batch, s, d, BLOCKSIZE, Wq, Wk, Wv);

  cudaDeviceSynchronize();

  printf("Q Projection Output:\n");
  for (int i = 0; i < s; i++) {
      for (int j = 0; j < d; j++) {
          printf("%.3f ", Q[i * d + j]);
      }
      printf("\n");
  }

  cudaFree(x);
  cudaFree(Wq);
  cudaFree(Wk);
  cudaFree(Wv);
  cudaFree(Q);
  cudaFree(K);
  cudaFree(V);

  return 0;
}

#include<stdio.h>
#include<cuda.h>

__global__ void kl_div(float *P, float *Q, float *loss , int N, int M, float eps){

  extern __shared__ float smem[];
  int row = blockIdx.x;
  if(row >= M) return;
  int idx = threadIdx.x;

  float local_loss = 0.0f;
  for(int i = idx ; i <N; i+=blockDim.x){
      float curr_p = P[row * N + i];
      float curr_q = Q[row *N + i ];
      local_loss += curr_p * (logf(fmaxf(curr_p , eps)) - log(curr_q));
   }

  smem[idx] = local_loss;
  __syncthreads();
  //reduction time
  for(int i = blockDim.x/2 ; i>0; i/=2){
    if(idx<i){
      smem[idx] += smem[idx + i];
    }
    __syncthreads();
  }
  if(idx==0){
    loss[row] = smem[0];
  }
  
}






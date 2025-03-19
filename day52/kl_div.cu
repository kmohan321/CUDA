#include<stdio.h>
#include<cuda.h>


/*
here P -> true or target probability (either log_probability or simply p so depends on lop_p)
Q -> predicted probability (should be log_probability i.e, given as Q = log(q))
*/
__global__ void kl_div(float *P, float *Q, float *loss , int N, int M, float eps, bool log_p){

  extern __shared__ float smem[];
  int row = blockIdx.x;
  if(row >= M) return;
  int idx = threadIdx.x;

  float local_loss = 0.0f;
  for(int i = idx ; i <N; i+=blockDim.x){
      float curr_p = P[row * N + i];
      float curr_q = Q[row *N + i ];
      if(!log_p){ //if p is not log(p) i.e, log_p is False
        local_loss += curr_p * (logf(fmaxf(curr_p , eps)) - curr_q);
      }
      else{
        local_loss += expf(curr_p) * (curr_p - curr_q);
      }
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
//computing the gradient with respect to the target
/*
grad_loss -> gradient of loss with respect to the output 
grad -> gradient of loss with respect to the target
*/
__global__ void kl_div_back(float *P,float *grad_loss,float *grad,int N, int M, bool log_p){

  extern __shared__ float smem[];
  int row = blockIdx.x;
  if(row >= M) return;
  int idx = threadIdx.x;

  for(int i = idx ; i <N; i+=blockDim.x){
      float curr_p = P[row * N + i];
      if(!log_p){ //if p is not log(p) i.e, log_p is False
        float curr_grad = curr_p * -1.0f;
        grad[row *N + i] = curr_grad * grad_loss[row * N + i];
      }
      else{
        float curr_grad = -expf(curr_p);
        grad[row *N + i] = curr_grad * grad_loss[row * N + i];
      }
   }
}







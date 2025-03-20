#include<stdio.h>
#include<cuda.h>

__global__ void geglu_forward(float *a, float *b, float *c, int M, int N){

  int row = blockIdx.x;
  int idx = threadIdx.x;
  if(row>=M) return;

  for(int i = idx; i <N ; i+=blockDim.x){
    float curr_a = a[row *N + i];
    float curr_b = b[row *N + i];
    
    float sqrt_2_over_pi = 0.7978845608028654;
    float a_3 = curr_a * curr_a *curr_a;
    float gelu_a = 0.5 * curr_a * (1 + tanhf(sqrt_2_over_pi * (curr_a + 0.044715 * a_3)));
    c[row *N + i] = gelu_a * curr_b;

  }
} 
__global__ void geglu_backward(float *grad_out, float *a, float *b,float *grad_a,float *grad_b,  int M ,int N){

  int row = blockIdx.x;
  int idx = threadIdx.x;
  if(row >=M) return;

  for(int i = idx; i <N ; i+=blockDim.x){
    float curr_a = a[row *N + i];
    float curr_b = b[row *N + i];
    float curr_grad_out = grad_out[row *N +i];
    
    float sqrt_2_over_pi = 0.7978845608028654;
    float a_3 = curr_a * curr_a *curr_a;
    float tanh_term = tanhf(sqrt_2_over_pi * (curr_a + 0.044715 * a_3));
    float gelu_a = 0.5 * curr_a * (1 + tanh_term);
    
    grad_b[row *N + i] =  curr_grad_out * gelu_a;

    float term1 = 0.5 * (1 + tanh_term);
    float tanh_sq = tanh_term * tanh_term;
    float term2 = 0.5 * curr_a * (1 - tanh_sq) * (sqrt_2_over_pi * (1 + 3 * 0.044715 * curr_a * curr_a));
    grad_a[row *N + i] =  curr_grad_out * curr_b * (term1 + term2);

  }
}



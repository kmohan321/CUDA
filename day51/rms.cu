#include<stdio.h>
#include<cuda.h>

__global__ void rms_norm (float *input, float *y, float *weight,float *Rstd, int M,int N,float eps){

    extern __shared__ float smem[];
    int row = blockIdx.x;
    int idx = threadIdx.x;
    if(row >= M) return ;

    float local_sqsum = 0.0f;
    for(int i = idx; i < N; i += blockDim.x){

      float x = input[row * N + i];
      local_sqsum += x*x;
    }
    smem[idx] = local_sqsum;
    __syncthreads();

    //reduction time
    for(int i = blockDim.x/2 ; i>0; i/=2){
      if(idx<i){
        smem[idx] += smem[idx + i];
      }
      __syncthreads();
    }

    float row_sqsum = smem[0] / N;
    float rstd = rsqrtf(row_sqsum + eps);

    //storing it for gradient computation
    Rstd[row] = rstd;

    //normalizing the values
    for(int i = idx; i < N; i += blockDim.x){

        float normalized = input[row * N + i] / rstd;
        y[row * N + i] = weight[i] * normalized;
      
    }
}

__global__ void rms_norm_back(float *dy, float *X, float *weight, float *Rstd,float *dx, float *dw,int M, int N ){

    int row = blockIdx.x;
    int idx = threadIdx.x;
    if(row >= M) return;

    extern __shared__ float smem[];
    float rstd =  Rstd[row];
    

    float local_factor = 0.0f;
    for(int i = idx ; i < N; i+=blockDim.x){

      float curr_dy = dy[row *N + i];
      float curr_x = X[row *N + i];
      float w = weight[i];
      dx[row * N + i] = rstd * curr_dy * w;

      local_factor += w * curr_dy *curr_x;
    }

    smem[idx] = local_factor;
    __syncthreads();
    for(int i = blockDim.x /2 ; i>0; i/=2){
      if(idx<i){
        smem[idx] += smem[idx + i];
      }
      __syncthreads();
    }
    float row_factor = smem[0]/N;

    for(int i = idx ; idx < N; i+=blockDim.x){
      float curr_x = X[row *N + i];
      dx[row *N + i] -= curr_x * rstd * rstd * row_factor;
    }
}


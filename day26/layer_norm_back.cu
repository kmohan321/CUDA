#include<stdio.h>
#include<cuda.h>

/*
-summation across batch
-launch threads equal to the nummber of d dimension
-each thread works on one d dimensional
-shape d_beta -> (D,)
-shape d_gamma -> (D,)
*/
__global__ void scale_grad (float *d_beta, float *d_gamma,float *dy, float *dx_hat,int N, int D){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx>=D) return;

    float gamma_sum = 0.f;
    float beta_sum = 0.f;

    for(int n =0; n<N; n++){
      int idn = n * D + idx;
      gamma_sum += dx_hat[idn] * dy[idn];
      beta_sum += dx_hat[idn];   
    }

    d_gamma[idx] = gamma_sum;
    d_beta[idx] = beta_sum;
}

/*
- gradient with respect to the normalised values
- y = gamma * x_hat + beta 
- d_y /d_x_hat = gamma
- overall loss => gamma * d_l/d_y(grad of loss wrt out y)
- let's launch threads equal to the number of rows (batch elements)
*/
__global__ void grad_norm (float *gamma, float *dy, float *dx_hat,int N,int D){

    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if(row>=N) return;

    for(int d =0 ;d<D; d++){
        dx_hat[row*D + d] = gamma[d]*dy[row*D+d];
    }

}

#define blocksize 32
__global__ void input_grad(float* dhatx, float* input, float* mean, float* var, float* dx, int N, int D, float epsilon){

    int row = blockIdx.x;
    int x = threadIdx.x;

    extern __shared__ float smem[];
    float *dhatx_sum = smem;
    float *dhatx_sum_xmu = smem + blocksize;

    float row_mean = mean[row];
    float sigma_inv = rsqrtf(var[row] + epsilon);

    float sum_dhatx = 0.0f;
    float sum_dhatx_xmu = 0.0f;

    for(int i = x ; i<D; i+= blocksize){
        float curr_value = dhatx[row* D + i];
        sum_dhatx += curr_value;
        sum_dhatx_xmu += curr_value * input[row*D+i];
    }
    dhatx_sum[x] = sum_dhatx;
    dhatx_sum_xmu[x] = sum_dhatx_xmu;
    __syncthreads();

    //reduction time
    for(int i = blocksize/2; i>0; i/=2){
      if(x<i){
        dhatx_sum[x] += dhatx_sum[x+i];
        dhatx_sum_xmu[x] += dhatx_sum_xmu[x+i];
      }
      __syncthreads();
    }

    float row_dhatx_sum= dhatx_sum[0];
    float row_dhatx_sum_xmu = dhatx_sum_xmu[0];

    for(int i = x ; i<D; i+= blocksize){
        float curr_value = dhatx[row* D + i];
        dx[row*D + i] = sigma_inv * (dhatx[row*D+i] - sum_dhatx / D - (input[row*D+i] - row_mean) * sum_dhatx_xmu * sigma_inv * sigma_inv / D); 
    }
    
}






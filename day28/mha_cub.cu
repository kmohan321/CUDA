#include<stdio.h>
#include<cuda.h>
#include <cublas_v2.h>

__global__ void soft_opt(float *input, float *output,int r, int c){

  extern __shared__ float smem[];
  
  int x = threadIdx.x;
  int row = blockIdx.x; 
  
  if(row>=r) return;
  
  float norm = 0.0f;
  float max_value = -INFINITY;
  for(int i = x; i<c; i += blockDim.x){ 

      float current_value = input[i + row*c];
      if(current_value > max_value){
          norm = norm * expf(max_value-current_value);
          max_value = current_value;
      }
      norm += expf(current_value-max_value);
  }
  __syncthreads();
  smem[x] = max_value;
  __syncthreads();
  
  //reduction time
  for(int stride = blockDim.x/2 ; stride>0; stride /=2 ){
      if(x<stride){
          smem[x] = fmax(smem[x],smem[x+stride]);
      }
      __syncthreads();
  }
  float global_maxvalue = smem[0];
  __syncthreads();
  //so now we have global_maxvalue, time for corrected norm
  norm *= expf(max_value-global_maxvalue);
  smem[x] = norm;
  __syncthreads();

  //reduction time for norm
  for(int stride = blockDim.x/2 ; stride>0; stride /=2 ){
      if(x < stride){
          smem[x] += smem[x+stride];
      }
      __syncthreads();
  }
  float final_norm = smem[0];
  __syncthreads();

  //time for softmax
  for(int i = x; i<c; i += blockDim.x){
      output[i + row*c] = expf(input[i + row*c]-global_maxvalue) / final_norm;
  }
  
}

int main(){

  cublasHandle_t handle;
  cublasCreate(&handle);

  int d = 100;
  int s = 24;
  int b = 16;

  //first performing projection

  float *x_d;
  float *x = (float*)malloc(b*s*d*sizeof(float));
  cudaMalloc((void**)&x_d, b*s*d*sizeof(float));
  cudaMemcpy(x_d,x,b*s*d*sizeof(float),cudaMemcpyHostToDevice);
  
  float *wq_d, *wk_d, *wv_d;
  float *wq = (float*)malloc(d*d*sizeof(float));
  float *wk = (float*)malloc(d*d*sizeof(float));
  float *wv = (float*)malloc(d*d*sizeof(float));

  cudaMalloc((void**)&wq_d,d*d*sizeof(float));
  cudaMalloc((void**)&wk_d,d*d*sizeof(float));
  cudaMalloc((void**)&wv_d,d*d*sizeof(float));

  cudaMemcpy(wq_d,wq,d*d*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(wk_d,wk,d*d*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(wv_d,wv,d*d*sizeof(float),cudaMemcpyHostToDevice);

  float *q_d, *k_d, *v_d;
  // float *q = (float*)malloc(b*s*d*sizeof(float));
  // float *k = (float*)malloc(b*s*d*sizeof(float));
  // float *v = (float*)malloc(b*s*d*sizeof(float));

  cudaMalloc((void**)&q_d,b*s*d*sizeof(float));
  cudaMalloc((void**)&k_d,b*s*d*sizeof(float));
  cudaMalloc((void**)&v_d,b*s*d*sizeof(float));

  float alpha = 1.0f, beta = 0.0f;

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
    d, b * s, d,
    &alpha, 
    wq_d, d,  
    x_d, d,   
    &beta, 
    q_d, d);  

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
      d, b * s, d, 
      &alpha, 
      wk_d, d, 
      x_d, d, 
      &beta, 
      k_d, d);

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
      d, b * s, d, 
      &alpha, 
      wv_d, d, 
      x_d, d, 
      &beta, 
      v_d, d);

  // cudaMemcpy(q, q_d, b * s * d * sizeof(float), cudaMemcpyDeviceToHost);
  // cudaMemcpy(k, k_d, b * s * d * sizeof(float), cudaMemcpyDeviceToHost);
  // cudaMemcpy(v, v_d, b * s * d * sizeof(float), cudaMemcpyDeviceToHost);
  
  float *S_d;
  float *S = (float*)malloc(b*s*s*sizeof(float));
  cudaMalloc((void**)&S_d,b*s*s*sizeof(float));
  
  cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,
    s,b*s,d,&alpha, q_d, d, k_d,d,&beta,
    S_d,s);

  // cudaMemcpy(S,S_d,b*s*s*sizeof(float),cudaMemcpyDeviceToHost);
  float *S_out;
  cudaMalloc((void**)&S_out, b*s*s*sizeof(float));

  dim3 grid(b*s);
  dim3 threads(256);
  int mem_size = 256*sizeof(float);
  soft_opt<<<grid,threads,mem_size>>>(S_d,S_out,b*s,s);

  float *o_d;
  cudaMalloc((void**)&o_d,b*s*d*sizeof(float));

  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
    d, b * s, s, 
    &alpha, 
    S_out, s, 
    v_d, s, 
    &beta, 
    o_d, d);

  cudaFree(x_d);
  cudaFree(wq_d); cudaFree(wk_d); cudaFree(wv_d);
  cudaFree(q_d); cudaFree(k_d); cudaFree(v_d);
  cudaFree(S_d); cudaFree(S_out); cudaFree(o_d);
  cublasDestroy(handle);

}
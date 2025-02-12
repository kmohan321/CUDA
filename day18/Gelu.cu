#include<stdio.h>
#include<cuda.h>

#define pi 3.14f

__global__ void forward(float *input, float *output, int N, int M){

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = gridDim.x * blockDim.x;

    float precomputed = sqrtf(2/pi);

    for (int i = idx ; i < M*N ; i+= stride){ //loads multiple elements
        float x = input[i];
        float x3 = x*x*x;
        float tanh_val = tanhf(precomputed * (x + 0.044715f * x3));
        output[i] = 0.5f * x * (1.0f + tanh_val);
    }
}

__global__ void backward(float *input,float *grad_input, float *output, int N, int M){

  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = gridDim.x * blockDim.x;

  float precomputed = sqrtf(2/pi);

  for (int i = idx ; i < M*N ; i+= stride){ //loads multiple elements

      float x = input[i];
      float x3 = x*x*x;

      float gx = precomputed*(x + 0.044715f * x3);
      float gx_prime = precomputed * (1.0f + 3*0.044715f*x*x);
      float tanh_gx = tanf(gx);
      
      float gelu_derv = 0.5f *(1.0f + tanh_gx + x*gx_prime * (1.0f - tanh_gx*tanh_gx));
      output[i] = grad_input[i] * gelu_derv;
  }
}





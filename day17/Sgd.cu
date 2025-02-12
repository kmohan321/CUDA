#include<stdio.h>
#include<cuda.h>


#define r 1000
#define c 1000
#define blocksize 32

__global__ void sgd (float *weight, float *grad, float lr){

      int x = threadIdx.x;
      int y  = threadIdx.y;

      int col = x + blockDim.x * blockIdx.x;
      int row = y + blockDim.y * blockIdx.y;

      if(row<r&& col <c){
        weight[col + row*c] -= lr * grad[col + row*c];
      }
}


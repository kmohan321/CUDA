#include<stdio.h>
#include<cuda.h>

#define tile 32

__global__ void fused_head_bce(){

    extern __shared__ float smem[];
  
    float *input_tiled = smem;
    float *weight_tiled = smem * tile*tile;

    for(int i = 0 ; i<)
}



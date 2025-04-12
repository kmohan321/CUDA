#include<cuda.h>
#include<stdio.h>


//we are going to handle one row per block
__global__ void quantised_weight_kernel(

  const float *weights,
  uint8_t *quantised_weight,
  float *scales,
  int M,
  int N
){

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int row_offset = row * N;

    extern __shared__ float smem[];

    float local_max = 0.0f;
    for(int i = tid; i < N; i += blockDim.x){
        local_max = fmaxf(0.0f, fabsf(weights[row_offset + i]));
    }

    smem[tid] = local_max;
    __syncthreads();

    for(int i = blockDim.x /2; i>0; i/=2){
      if(tid<i){
        smem[tid] = fmaxf(smem[tid + i], smem[tid]);
      }
      __syncthreads();
    }

    float scale = 1.0f;
    if (tid == 0) {
        scale = smem[0] / 7.0f; // symmetric int4: [-8, 7]
        scale = fmaxf(scale, 1e-8f);
        scales[row] = scale;
    }

    __syncthreads();

    for (int i = tid * 2; i < N; i += blockDim.x * 2) {
      int8_t q0 = 0, q1 = 0;
      if (i < N) {
          float val0 = weights[row_offset + i];
          int q = __float2int_rn(val0 / scale);
          q = max(-8, min(7, q));
          q0 = (int8_t)(q + 8); // store in [0, 15]
      }
      if (i + 1 < N) {
          float val1 = weights[row_offset + i  + 1];
          int q = __float2int_rn(val1 / scale);
          q = max(-8, min(7, q));
          q1 = (int8_t)(q + 8); // store in [0, 15]
      }

      uint8_t packed = (q0 << 4) | (q1 & 0x0F);
      if (i / 2 < (N + 1) / 2) {
        quantised_weight[row_offset + (i / 2)] = packed;
      }
  }

}
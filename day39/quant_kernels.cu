#include<stdio.h>
#include<cuda.h>


//each block handles one row for efficiency
//first load quantised weights
__device__ __forceinline__ void unpack_int4(int8_t packed, int8_t &low, int8_t &high) {
  low = packed & 0x0F;       
  high = (packed >> 4) & 0x0F; 

  low = (low >= 8) ? (low - 16) : low;
  high = (high >= 8) ? (high - 16) : high;
}

__global__ void quantised_softmax(int8_t *input, float *output, int M, int N, float *scales){

    int row = blockIdx.x;
    int x = threadIdx.x;

    float local_sum = 0.0f;
    float local_max = -INFINITY;

    float scale = scales[row];

    extern __shared__ float smem[];
    float *r_max = smem;
    float *r_sum = smem + blockDim.x;

    for(int j = x; j < N; j += blockDim.x){

        int8_t packed = input[row * N + j];
        int8_t curr_value, next_curr_value;
        unpack_int4(packed,curr_value,next_curr_value);

        float dequantized1 = scale * curr_value;
        float dequantized2 = scale * next_curr_value;

        if(dequantized1 > local_max){
          local_sum = local_sum * expf(local_max - dequantized1);
          local_max = dequantized1;
        }
        local_sum += expf(dequantized1 - local_max);
        if(dequantized2 > local_max){
          local_sum = local_sum * expf(local_max - dequantized2);
          local_max = dequantized2;
        }
        local_sum += expf(dequantized2 - local_max);
    }
    r_max[x] = local_max;
    __syncthreads();

    for(int i = blockDim.x/2 ; i>0 ; i/=2){
        if(x<i){
          r_max[x] = fmaxf(r_max[x+i],r_max[x]);
        }
        __syncthreads();
    }
    float row_max = r_max[0];
    r_sum[x] =  local_sum * expf(local_max - row_max);
    __syncthreads();

    for(int i = blockDim.x/2; i>0; i/=2){
        if(x < i){
          r_sum[x] += r_sum[x+i];
        }
        __syncthreads();
    }
    float row_norm = r_sum[0];

    for(int j = x ; j < N; j+=blockDim.x){

      int8_t packed = input[row * N + j];
      int8_t curr_value, next_curr_value;
      unpack_int4(packed,curr_value,next_curr_value);

      float dequantized1 = scale * curr_value;
      float dequantized2 = scale * next_curr_value;
      output[row * N + 2*j] = expf(dequantized1-row_max) / row_norm;
      output[row * N + 2*j + 1] = expf(dequantized2-row_max) / row_norm;
    }

}
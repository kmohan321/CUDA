#include<stdio.h>
#include<cuda.h>

/*
x -> shape(r,k) -> float32 weights
w -> shape(k,c) -> qunatised int4
out -> shape(r,c) -> float32
k -> common dimension
*/
#define tile 32
#define R 256  // Number of rows in x
#define K 128 // Common dimension
#define C 126  // Number of columns in w

__global__ void quant_matmul(float *out, float *x, uint8_t *w, int *s, int *z, int r, int k, int c){

    extern __shared__ float smem[];
    float *tile_x = smem;
    float *tile_w = tile_x + tile*tile;

    int idx = threadIdx.x;
    int idy = threadIdx.y;
    int col = idx + blockDim.x * blockIdx.x;
    int row = idy + blockDim.y * blockIdx.y;

    int scale = (col < c) ? s[col] : 1; //per row z and s 
    int zero_point = (col < c) ? z[col] : 0;

    float sum = 0.0f;
    for(int i = 0 ; i < (k + tile - 1)/tile; i++){

        //loading the input in shared memory
        if(row < r && (idx + i*tile) < k){
          tile_x[idx + idy*tile] = x[row * k + idx + i*tile];
        }
        else{
          tile_x[idx + idy*tile] = 0.0f;
        }
        __syncthreads();
        //real work
        //unpacking and dequantisation of weights
        int s_tile = tile/2;
        if((idx + i * s_tile) < k && col < c){
          uint8_t packed_w = w[(col*k/2) + idx/2  + i * s_tile];
          //holding values 4 bit unsinged int into int 
          int w1 = packed_w & 0x0F;  // Lower 4 bits -> (0x0F -> binary_mask)
          int w2 = (packed_w >> 4) & 0x0F;  // Upper 4 bits

          float weight1 = scale * (w1 - zero_point);
          float weight2 = scale * (w2 - zero_point); 

          tile_w[idy * tile + 2*idx] = weight1;
          tile_w[idy * tile + 2*idx + 1] = weight2;
        } 
        else{
          tile_w[idy * tile + 2*idx] = 0.0f;
          tile_w[idy * tile + 2*idx +1] = 0.0f;
        }
        __syncthreads();

        for(int j = 0; j<tile; j++){
            sum += tile_x[idy * tile + j] * tile_w[idx + tile*j];
        }
        __syncthreads();
    }
    if(row<r && col<c){
      out[row * c + col] = sum;
    }
    
}

//quantize float32 weights to int4 (packed into uint8)
void quantize_weights(float *w, uint8_t *qw, int *s, int *z, int c, int k) {
  for (int j = 0; j < c; j++) {
      float max_val = -1e10, min_val = 1e10;

      for (int i = 0; i < k; i++) {
          float val = w[i * c + j];  
          if (val > max_val) max_val = val;
          if (val < min_val) min_val = val;
      }

      // s[j] = (max_val - min_val) / 15.0f; //for symmertric 
      // z[j] = min_val;
      s[j] = max_val / 15.0f; //for asymmetric 
      z[j] = 0.0f;

      for (int i = 0; i < k; i += 2) {
          int w1 = (w[i * c + j] - z[j]) / s[j]; 
          int w2 = (i + 1 < k) ? (w[(i + 1) * c + j] - z[j]) / s[j] : 0;

          w1 = max(0, min(15, w1));
          w2 = max(0, min(15, w2));
          qw[j * (k / 2) + i / 2] = (w2 << 4) | w1; //shape -> (c,k/2)
      }
  }
}

void matmul_cpu(float *out, float *x, float *w, int r, int k, int c) {
  for (int i = 0; i < r; i++) {  
      for (int j = 0; j < c; j++) {  
          float sum = 0.0f;
          for (int t = 0; t < k; t++) { 
              sum += x[i * k + t] * w[t * c + j];  
          }
          out[i * c + j] = sum;
      }
  }
}
void compare_results(float *cpu_out, float *gpu_out, int r, int c) {
  float error_sum = 0.0f;
  float max_error = 0.0f;
  for (int i = 0; i < r * c; i++) {
      float error = fabs(cpu_out[i] - gpu_out[i]);
      error_sum += error;
      max_error = fmax(max_error, error);
  }
  float mean_error = error_sum / (r * c);
  
  printf("Mean Absolute Error : %.2f\n",mean_error);
  printf("Max Error : %.2f\n",max_error);
}

int main() {

    float *h_x = (float *)malloc(R * K * sizeof(float));
    float *h_out = (float *)malloc(R * C * sizeof(float));
    float *h_w = (float *)malloc(K * C * sizeof(float));
    float *cpu_out = (float *)malloc(R * C * sizeof(float));

    uint8_t *h_qw = (uint8_t *)malloc((K * C) / 2 * sizeof(uint8_t));
    int *h_s = (int *)malloc(K * sizeof(int));
    int *h_z = (int *)malloc(K * sizeof(int));

    for (int i = 0; i < R * K; i++) h_x[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    for (int i = 0; i < K * C; i++) h_w[i] = ((float)rand() / RAND_MAX) * 2 - 1;

    //cpu matmul(without quantisation)
    matmul_cpu(cpu_out, h_x, h_w,R,K,C);

    quantize_weights(h_w, h_qw, h_s, h_z, K, C);

    float *d_x, *d_out;
    uint8_t *d_qw;
    int *d_s, *d_z;

    cudaMalloc((void **)&d_x, R * K * sizeof(float));
    cudaMalloc((void **)&d_out, R * C * sizeof(float));
    cudaMalloc((void **)&d_qw, (K * C) / 2 * sizeof(uint8_t));
    cudaMalloc((void **)&d_s, K * sizeof(int));
    cudaMalloc((void **)&d_z, K * sizeof(int));

    cudaMemcpy(d_x, h_x, R * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qw, h_qw, (K * C) / 2 * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, h_s, K * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, K * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blockDim(tile, tile);
    dim3 gridDim((C + tile - 1) / tile, (R + tile - 1) / tile);
    size_t sharedMemSize = 2 * tile * tile * sizeof(float);

    quant_matmul<<<gridDim, blockDim, sharedMemSize>>>(d_out, d_x, d_qw, d_s, d_z, R, K, C);

    cudaMemcpy(h_out, d_out, R * C * sizeof(float), cudaMemcpyDeviceToHost);

    compare_results(cpu_out,h_out,R,C);
    cudaFree(d_x);
    cudaFree(d_out);
    cudaFree(d_qw);
    cudaFree(d_s);
    cudaFree(d_z);

    free(h_x);
    free(h_out);
    free(h_w);
    free(h_qw);
    free(h_s);
    free(h_z);
    free(cpu_out);

    printf("Computation Completed Successfully!\n");
    return 0;
}

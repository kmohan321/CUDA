#include<stdio.h>
#include<cuda.h>

#define h 100
#define w 100
#define C 3 
#define tile 16

__global__ void ddpm(float *noise, float *x_t, float*x_t_1, float noise_signal, float image_signal){

    int x = threadIdx.x;
    int y = threadIdx.y;
    int col = x  + blockDim.x * blockIdx.x;
    int row = y + blockDim.y * blockIdx.y; 

    int c = blockIdx.z;
    int _offset = c*(h*w);
    int tile_offset = c*(tile*tile);

    __shared__ float noised_image[tile*tile*C];
    __shared__ float noise_tile[tile*tile*C];

    if(row<h && col<w){
      noised_image[tile_offset + x + y*tile] = x_t[_offset + row * w + col];
      noise_tile[tile_offset + x + y*tile] = noise[_offset + row * w + col];
    }
    __syncthreads();

    if(row<h && col<w){
      x_t_1[_offset + row * w + col] =  image_signal * noised_image[tile_offset + x + y*tile] + noise_signal *noise_tile[tile_offset + x + y*tile];
    }
    
}

__global__ void ddpm_denoise(float *x_t, float *noise_pred, float *x_t_1, float alpha_t, float alpha_bar_t, float sigma_t){

    int x = threadIdx.x;
    int y = threadIdx.y;
    int col = x  + blockDim.x * blockIdx.x;
    int row = y + blockDim.y * blockIdx.y; 
    int c = blockIdx.z;

    int _offset = c * (h * w);
    int tile_offset = c * (tile * tile);

    __shared__ float img_tile[tile*tile*C];
    __shared__ float noise_tile[tile*tile*C];

    if(row < h && col < w){
        img_tile[tile_offset + x + y*tile] = x_t[_offset + row * w + col];
        noise_tile[tile_offset + x + y*tile] = noise_pred[_offset + row * w + col];
    }
    __syncthreads();

    if(row < h && col < w){
        float mean = (1.0f / sqrtf(alpha_t)) * (img_tile[tile_offset + x + y*tile] - ((1 - alpha_t) / sqrtf(1 - alpha_bar_t)) * noise_tile[tile_offset + x + y*tile]);
        float z = sigma_t * noise_tile[tile_offset + x + y*tile]; 
        x_t_1[_offset + row * w + col] = mean + z;
    }
}


int main() {
  float *h_x_t, *h_noise, *h_x_t_1;
  float *d_x_t, *d_noise, *d_x_t_1; 

  int size = h * w * C * sizeof(float);

  h_x_t = (float*)malloc(size);
  h_noise = (float*)malloc(size);
  h_x_t_1 = (float*)malloc(size);

  for (int i = 0; i < h * w * C; i++) {
      h_x_t[i] = rand() / (float)RAND_MAX;  // Random values between 0 and 1
      h_noise[i] = rand() / (float)RAND_MAX;
  }

  cudaMalloc(&d_x_t, size);
  cudaMalloc(&d_noise, size);
  cudaMalloc(&d_x_t_1, size);

  cudaMemcpy(d_x_t, h_x_t, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_noise, h_noise, size, cudaMemcpyHostToDevice);

  dim3 blockDim(tile, tile);
  dim3 gridDim((w + tile - 1) / tile, (h + tile - 1) / tile, C);

  float noise_signal = 0.1f;
  float image_signal = 0.9f;

  ddpm<<<gridDim, blockDim>>>(d_noise, d_x_t, d_x_t_1, noise_signal, image_signal);
  cudaDeviceSynchronize();

  cudaMemcpy(h_x_t_1, d_x_t_1, size, cudaMemcpyDeviceToHost);

  printf("Sample output:\n");
  for (int i = 0; i < 5; i++) {
      printf("x_t_1[%d] = %f\n", i, h_x_t_1[i]);
  }
  
  FILE *f_noise = fopen("noise.raw", "wb");
  fwrite(h_noise, sizeof(float), h * w * C, f_noise);
  fclose(f_noise);


  free(h_x_t);
  free(h_noise);
  free(h_x_t_1);
  cudaFree(d_x_t);
  cudaFree(d_noise);
  cudaFree(d_x_t_1);

  return 0;
}
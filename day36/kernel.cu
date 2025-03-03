#include<stdio.h>
#include<cuda.h>

__constant__ float precomputed_sin[512];  // sin values
__constant__ float precomputed_cos[512];  // cos values


__global__ void RoPE(float *input , float *output, float *freq, int d, int N){

    int batch = blockIdx.x;
    int sequence = blockIdx.y;
    int x = threadIdx.x;

    if(x < d/2){
      float sin_theta = precomputed_sin[x];
      float cos_theta = precomputed_cos[x];

      float curr_value = input[batch *(N*d) + sequence *d + 2* x];
      float next_value = input[batch *(N*d) + sequence *d + 2* x +1];
      float x_ = curr_value * cos_theta + next_value * sin_theta;
      float y_ = curr_value * sin_theta - cos_theta * next_value;
  
      output[batch * (N*d) + sequence *d +  2 * x] = x_;
      output[batch * (N*d) + sequence *d + 2* x +1] = y_;
    }
}

int main() {
  int batch_size = 1; 
  int N = 4;           
  int d = 8;           

  float *h_input, *h_output, *h_freq;
  h_input = (float*) malloc(batch_size * N * d * sizeof(float));
  h_output = (float*) malloc(batch_size * N * d * sizeof(float));
  h_freq = (float*) malloc(d/2 * sizeof(float));

  for(int i = 0; i < batch_size * N * d; i++) {
      h_input[i] = (float)(i + 1);
  }

  for(int i = 0; i < d/2; i++) {
      h_freq[i] = powf(10000.0f, -2.0f * i / d);
  }

  float sin_values[512], cos_values[512];
  for (int i = 0; i < d/2; i++) {
      float theta =  h_freq[i];
      sin_values[i] = sinf(theta);
      cos_values[i] = cosf(theta);
  }

  cudaMemcpyToSymbol(precomputed_sin, sin_values, (d/2) * sizeof(float));
  cudaMemcpyToSymbol(precomputed_cos, cos_values, (d/2) * sizeof(float));

  float *d_input, *d_output, *d_freq;
  cudaMalloc((void**)&d_input, batch_size * N * d * sizeof(float));
  cudaMalloc((void**)&d_output, batch_size * N * d * sizeof(float));
  cudaMalloc((void**)&d_freq, d/2 * sizeof(float));

  cudaMemcpy(d_input, h_input, batch_size * N * d * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_freq, h_freq, d/2 * sizeof(float), cudaMemcpyHostToDevice);

  dim3 grid(batch_size, N); 
  dim3 block(d/2);          

  RoPE<<<grid, block>>>(d_input, d_output, d_freq, d, N);
  cudaDeviceSynchronize();

  cudaMemcpy(h_output, d_output, batch_size * N * d * sizeof(float), cudaMemcpyDeviceToHost);

  printf("Input:\n");
  for(int i = 0; i < batch_size * N * d; i++) {
      printf("%.3f ", h_input[i]);
      if ((i + 1) % d == 0) printf("\n");
  }

  printf("\nOutput after RoPE:\n");
  for(int i = 0; i < batch_size * N * d; i++) {
      printf("%.3f ", h_output[i]);
      if ((i + 1) % d == 0) printf("\n");
  }

  free(h_input);
  free(h_output);
  free(h_freq);
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_freq);

  return 0;
}
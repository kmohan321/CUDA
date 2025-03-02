#include<stdio.h>
#include<cuda.h>

__global__ void spmm_csr_kernel(float *values, int *col_idx, int *row_ptr,
                                float *dense, float *output, int rows, int cols, int dense_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        for (int j = 0; j < dense_cols; j++) {
            float sum = 0.0f;
            for (int i = row_ptr[row]; i < row_ptr[row + 1]; i++) {
                sum += values[i] * dense[col_idx[i] * dense_cols + j];
            }
            output[row * dense_cols + j] = sum;
        }
    }
}


void spmm_csr(float *h_values, int *h_col_idx, int *h_row_ptr,
  float *h_dense, float *h_output, int rows, int cols, int dense_cols) {

  float *d_values, *d_dense, *d_output;
  int *d_col_idx, *d_row_ptr;
  cudaMalloc(&d_values, h_row_ptr[rows] * sizeof(float));
  cudaMalloc(&d_col_idx, h_row_ptr[rows] * sizeof(int));
  cudaMalloc(&d_row_ptr, (rows + 1) * sizeof(int));
  cudaMalloc(&d_dense, cols * dense_cols * sizeof(float));
  cudaMalloc(&d_output, rows * dense_cols * sizeof(float));

  cudaMemcpy(d_values, h_values, h_row_ptr[rows] * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_idx, h_col_idx, h_row_ptr[rows] * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_row_ptr, h_row_ptr, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_dense, h_dense, cols * dense_cols * sizeof(float), cudaMemcpyHostToDevice);

  int block_size = 256;
  int grid_size = (rows + block_size - 1) / block_size;
  spmm_csr_kernel<<<grid_size, block_size>>>(d_values, d_col_idx, d_row_ptr, d_dense, d_output, rows, cols, dense_cols);

  cudaMemcpy(h_output, d_output, rows * dense_cols * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_values);
  cudaFree(d_col_idx);
  cudaFree(d_row_ptr);
  cudaFree(d_dense);
  cudaFree(d_output);
}

int main() {
  float h_values[] = {1, 2, 3, 4, 5}; 
  int h_col_idx[] = {0, 2, 1, 0, 2};
  int h_row_ptr[] = {0, 2, 3, 5};    


  float h_dense[3][2] = {{1, 2}, {3, 4}, {5, 6}};

  float h_output[6] = {0};

  spmm_csr(h_values, h_col_idx, h_row_ptr, (float*)h_dense, h_output, 3, 3, 2);

  printf("Output Matrix:\n");
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      printf("%f ", h_output[i * 2 + j]);
      }
    printf("\n");
  }
  return 0;
}
#include<stdio.h>
#include<cuda.h>

#define TILE_SIZE 16  

__global__ void matrix_mul(float *m1, float *m2, float *m3, int tile, int r, int c, int common){

  int idx = threadIdx.x + blockDim.x * blockIdx.x; //column index
  int idy = threadIdx.y + blockDim.y * blockIdx.y; //row index
  int x = threadIdx.x;
  int y = threadIdx.y;
  
  extern __shared__ float smem[];
  float *s1 = smem;
  float *s2 = s1 + tile*tile;

  float sum = 0.0f; 
  for(int j =0;j<(common + tile -1)/tile;j++){

        if(idy <r && (x+j*tile)<common){
              s1[x + y*tile] = m1[idy * common + x + j*tile];
        }
        else{
          s1[x + y*tile] = 0.0f;
        }
        __syncthreads();

        if((y+tile*j)<common && idx<c){
              s2[x + y*tile] = m2[idx + y *c + c*tile*j]; 
        }
        else{
          s2[x + y*tile]=0.0f;
        }
        __syncthreads();

        if(idx<c && idy<r){
              for(int i =0 ; i<tile;i++){
                    sum += s1[i + y*tile] * s2[x + i*tile];
              }
        }
        __syncthreads();
  }
  if(idx<c && idy<r){
        m3[idx + idy*c] =sum;
  }
}

__global__ void swish(float *input, float *output, int tile, int r, int c){

  extern __shared__ float smem[];

  int col = threadIdx.x + blockDim.x * blockIdx.x; //column index
  int row = threadIdx.y + blockDim.y * blockIdx.y; //row index
  int x = threadIdx.x;
  int y = threadIdx.y;

  if(row<r && col <c){
      smem[x + y*tile] = input[row*c + col];
  }
  __syncthreads();

  if(row<r && col<c){
    float curr_value = smem[x+y*tile]; 
    output[row*c + col] = output[row*c + col] = curr_value * (1.0f / (1.0f + expf(-curr_value)));
  }

}

__global__ void swiglu(float *input1 ,float *input2, float *output, int r, int c){

  int col = threadIdx.x + blockDim.x * blockIdx.x; //column index
  int row = threadIdx.y + blockDim.y * blockIdx.y; //row index

  if(row<r && col<c){
    output[row*c + col] = input1[row*c + col] * input2[row*c + col];
  }

}

void initialize_matrix(float *matrix, int size) {
  for (int i = 0; i < size; i++) {
      matrix[i] = (float)rand() / (float)RAND_MAX;
  }
}

int main() {
    int r = 32, c = 32, common = 32;  
    int size_A = r * common * sizeof(float);
    int size_B = common * c * sizeof(float);
    int size_C = r * c * sizeof(float);

    float *h_A, *h_B, *h_C, *h_swiglu_out;
    float *d_A, *d_B, *d_C, *d_swiglu_out;

    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C = (float*)malloc(size_C);
    h_swiglu_out = (float*)malloc(size_C);

    initialize_matrix(h_A, r * common);
    initialize_matrix(h_B, common * c);

    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);
    cudaMalloc((void**)&d_swiglu_out, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((c + TILE_SIZE - 1) / TILE_SIZE, (r + TILE_SIZE - 1) / TILE_SIZE);

    int sharedMemSize = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
    matrix_mul<<<gridDim, blockDim, sharedMemSize>>>(d_A, d_B, d_C, TILE_SIZE, r, c, common);
    cudaDeviceSynchronize();

    float *input2;
    float *d_c_ = (float*)malloc(r*c*sizeof(float));
    cudaMemcpy(d_c_,d_C,r*c*sizeof(float),cudaMemcpyDeviceToHost); 
    
    swish<<<gridDim, blockDim, TILE_SIZE * TILE_SIZE * sizeof(float)>>>(d_C, d_C, TILE_SIZE, r, c);
    cudaDeviceSynchronize();

    cudaMalloc((void**)&input2,r*c*sizeof(float));
    cudaMemcpy(input2,d_c_,r*c*sizeof(float),cudaMemcpyHostToDevice); 
    swiglu<<<gridDim, blockDim>>>(input2, d_C, d_swiglu_out, r, c);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_swiglu_out, d_swiglu_out, size_C, cudaMemcpyDeviceToHost);

    printf("Output of SwiGLU (first 5 values):\n");
    for (int i = 0; i < 5; i++) {
        printf("%f ", h_swiglu_out[i]);
    }
    printf("\n");

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_swiglu_out);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_swiglu_out);

    return 0;
}

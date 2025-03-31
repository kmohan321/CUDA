#include<stdio.h>
#include<cuda.h>

__global__ void matmul(float *A, float *B, float *out, int M, int N,int K,int tile, bool is_sigmoid){

      int x = threadIdx.x;
      int y = threadIdx.y;
      int col = x + blockDim.x * blockIdx.x;
      int row = y + blockDim.y * blockIdx.y;
      
      extern __shared__ float smem[];

      float *tile_A = smem;
      float *tile_B = smem + tile*tile;

      //move over the tiles
      float sum = 0.0f;
      for(int tile_id = 0; tile_id<(K+tile-1)/tile; tile_id++){

        //loading the tiles in shared memory (using the threads to load parallely)
        if(row < M && (x + tile_id * tile) <K){
          tile_A[y * tile + x] = A[row * K + (x + tile_id * tile)];
        }
        else{
          tile_A[y*tile + x] = 0.0f;
        }

        if(col < N && (y + tile_id * tile)<K){
          tile_B[y * tile + x] = B[(y + tile_id*tile)*N + col]; //row *stride + col
        }
        else{
          tile_B[y * tile + x] = 0.0f;
        }
        __syncthreads();

        //computing the local sum
        for(int k = 0; k<tile; k++){
          sum += tile_A[y * tile + k] * tile_B[k * tile + x];
        }
        __syncthreads();
      }
      if(row < M && col <N){
        if(is_sigmoid){
          out[row *N + col] = 1.0f / (1.0f + expf(-sum));
        }
        else{
          out[row *N + col] = sum;
        }
      }
}

// Parallel Scan for Recurrence: h_t = (1 - z_t) h_{t-1} + z_t h_tilde
__global__ void parallel_scan(float *z, float *h_tilde, float *h_out, int batch_size, int seq_len, int hidden_size) {
  int batch = blockIdx.x;
  int hidden = threadIdx.x;

  if (hidden >= hidden_size) return;

  float h_prev = h_out[batch * hidden_size  + hidden]; // Initial h_0

  for (int t = 0; t < seq_len; t++) {
      int idx = batch * seq_len * hidden_size + t * hidden_size + hidden;
      float z_t = z[idx];
      float h_tilde_t = h_tilde[idx];

      h_prev = (1 - z_t) * h_prev + z_t * h_tilde_t;
      h_out[idx] = h_prev;  // Store the updated hidden state
  }
}

void initialize_matrix(float *matrix, int rows, int cols) {
  for (int i = 0; i < rows * cols; i++) {
      matrix[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Random values between -1 and 1
  }
}

void print_matrix(float *matrix, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
          printf("%0.2f ", matrix[i * cols + j]);
      }
      printf("\n");
  }
}

int main(){

    int b = 2, s = 4, h = 8;
    int input_size = 4;
    float *w_h = (float*)malloc(input_size * h * sizeof(float));
    float *w_z = (float*)malloc(input_size * h * sizeof(float));
    float *h_0 = (float*)malloc(b * 1 * h * sizeof(float));
    float *x = (float*)malloc(b * s * input_size * sizeof(float));

    float *d_wh, *d_wz, *d_x, *d_z, *d_htilde;
    cudaMalloc((void**)&d_wh,input_size*h*sizeof(float));
    cudaMalloc((void**)&d_wz,input_size*h*sizeof(float));
    cudaMalloc((void**)&d_x,b *s*input_size*sizeof(float));
    cudaMalloc((void**)&d_htilde,b*s*h*sizeof(float));
    cudaMalloc((void**)&d_z,b*s*h*sizeof(float));

    initialize_matrix(w_h, input_size, h);
    initialize_matrix(w_z, input_size, h);
    initialize_matrix(h_0, b, h);
    initialize_matrix(x, b * s, input_size);

    cudaMemcpy(d_wh, w_h, input_size*h*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_wz, w_z, input_size*h*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_x , x, b *s * input_size*sizeof(float),cudaMemcpyHostToDevice);

    //x @ w_h = h_tilde -> (b,s,input) @ (input, h) -> (b,s,h)
    //x @ w_z = z -> (b,s,input) @ (input, h) -> (b,s,h)
    int threads = 16;
    dim3 grid ((h + threads -1) / threads, (b*s + threads -1)/threads);
    dim3 blocksize(threads,threads);
    int smem_size = 2 * threads * threads *sizeof(float);
    //calling the matrix kernel
    matmul<<<grid , blocksize, smem_size>>>(d_x, d_wz,d_z, b*s ,h, input_size,threads,true);
    matmul<<<grid , blocksize, smem_size>>>(d_x, d_wh,d_htilde, b*s ,h, input_size,threads,false);

    float *d_hout;
    cudaMalloc((void**)&d_hout, b * s * h * sizeof(float));
    cudaMemset(d_hout, 0, b * s * h * sizeof(float));
    cudaMemcpy(d_hout, h_0, b * h * sizeof(float), cudaMemcpyHostToDevice);  // Initialize h_out with h_0

    dim3 grid_scan(b, 1, 1);
    int hidden_threads = h; //h should always be less than the 1024 threads
    parallel_scan<<<grid_scan, hidden_threads>>>(d_z, d_htilde, d_hout, b, s, h);

    float *h_out = (float*)malloc(b * s * h * sizeof(float));
    cudaMemcpy(h_out, d_hout, b * s * h * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    printf("Final hidden states:\n");
    for (int batch = 0; batch < b; batch++) {
        printf("Batch %d:\n", batch);
        for (int t = 0; t < s; t++) {
            printf("  t=%d: ", t);
            for (int hidden = 0; hidden < h; hidden++) {
                printf("%f ", h_out[batch * s * h + t * h + hidden]);
            }
            printf("\n");
        }
    }

    // Free memory
    cudaFree(d_wh);
    cudaFree(d_wz);
    cudaFree(d_x);
    cudaFree(d_z);
    cudaFree(d_htilde);
    cudaFree(d_hout);
    free(w_h);
    free(w_z);
    free(h_0);
    free(x);
    free(h_out);

}

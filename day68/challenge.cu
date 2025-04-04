#include <cuda_runtime.h>

__global__ void smem_matmul(const float *A, const float *B, float *out, int M, int N,int K,int tile, float scaling_factor,const float *bias){

  int x = threadIdx.x;
  int y = threadIdx.y;
  int col = x + blockDim.x * blockIdx.x;
  int row = y + blockDim.y * blockIdx.y;
  
  extern __shared__ float smem[]; //for caching the data 

  float *tile_A = smem;
  float *tile_B = smem + tile*tile;

  //move over the tiles
  float sum = 0.0f;
  for(int tile_id = 0; tile_id < (K+tile-1)/tile; tile_id++){

    //loading the tiles in shared memory (using the threads to load parallely)
    if(row < M && (x + tile_id * tile) <K){
      tile_A[y * tile + x] = A[row * K + (x + tile_id * tile)];
    }
    else{
      tile_A[y*tile + x] = 0.0f;
    }

    if(col < N && (y + tile_id * tile)<K){
      // tile_B[y * tile + x] = B[(y + tile_id*tile)*N + col]; //row *stride + col
      tile_B[y * tile + x] = B[col * K + (y + tile_id * tile)]; //simply transposing the matrix but not coalesced
    }
    else{
      tile_B[y * tile + x] = 0.0f;
    }
    __syncthreads();

    //computing the local sum
    for(int k = 0; k < tile; k++){
      sum += tile_A[y * tile + k] * tile_B[k * tile + x];
    }
    __syncthreads();
  }

  if(row < M && col <N){
    float output = sum + bias[col];
    out[row *N + col] = scaling_factor * output * (1.0f / (1.0f + expf(-output)));
  }
}


// Note: input_matrix, weight_matrix, bias, output are all device pointers to float32 arrays
extern "C" void solution(const float* input_matrix, const float* weight_matrix, const float* bias, float scaling_factor, float* output, size_t batch_size, size_t in_features, size_t out_features) {    

    int M = batch_size;
    int N = out_features;
    int K = in_features;
    int tile = 32;
    dim3 blocksize (tile, tile);
    dim3 grid ((N + tile -1)/tile, (M + tile -1)/tile);
    int smem_size = 2 * tile * tile * sizeof(float);
    smem_matmul<<<grid,blocksize,smem_size>>>(input_matrix,weight_matrix,output,M,N,K,tile,scaling_factor,bias);

}

__global__ void layernorm(const float *X, const float *gamma, const float *beta, float eps, float *Y, int f, int d1, int d2, int B,int N ){

    int row = blockIdx.x;
    int x = threadIdx.x;
    int batch_offset = row * f *d1 *d2;

    if(row >= B) return;
    extern __shared__ float smem[];
    float *s1_shared = smem;
    float *s2_shared = s1_shared + blockDim.x;

    float sum = 0.0f;
    float sq_sum = 0.0f;
    for(int i = x; i < N; i += blockDim.x){
        float curr_value = X[batch_offset + i];
        sum += curr_value;
        sq_sum += curr_value * curr_value;
    }
    s1_shared[x] = sum;
    s2_shared[x] = sq_sum;

    for(int i = blockDim.x/2; i > 0; i/=2){
        if(x < i){
            s1_shared[x] += s1_shared[x+i];
        }
        __syncthreads();
    }
    float mean = s1_shared[0]/N;
    float variance = (s2_shared[0] / N) - mean *mean;
    float r_variance = 1.0f / (sqrtf(variance + eps));
    for(int i = x; i < N; i += blockDim.x){
        float curr_value = X[batch_offset + i];
        float out = (curr_value - mean) * r_variance * gamma[i] + beta[i];
        Y[batch_offset + i] = out;
    }


}

// Note: X, gamma, beta, Y are all device pointers to float32 arrays
extern "C" void solution(const float* X, const float* gamma, const float* beta, float* Y, size_t B, size_t F, size_t D1, size_t D2) {    

    int M = B;
    int N =  F*D1*D2;
    int threads = 256;
    dim3 blocksize(256);
    float eps = 1e-5;
    dim3 grid(M);
    int smem_size = 2* threads *threads *sizeof(float);
    layernorm<<< grid, blocksize, smem_size>>>(X,gamma,beta,eps,Y,F,D1,D2,B,N);

}
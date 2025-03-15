#include<stdio.h>
#include<cuda.h>
#include <torch/types.h>


#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    }

//rememeber inputs are 4d tensors -> (b,h,s,d)
//B_r -> total elements in each query tile
//B_c -> total elements in each key and value tile  
__global__ void flash_attn(float *K, float *V, float *Q,float *O, float *l, float *m,
    int T_r, int T_c, int b, int h, int s, int d,int B_c, int B_r, float scale){
 
    extern __shared__ float smem[];

    float *k = smem;
    float *v = k + B_c * d;
    float *q = v +  B_c * d;
    float *S = q + B_r * d;

    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int x = threadIdx.x;

    //shared memory for local values
    float *local_max = S + B_r*B_c;  //B_r
    float *local_sum = local_max + B_r; //B_r
    float *row_max = local_sum + B_r;
    float *row_sum = row_max + B_r;
    float *prev_max = row_sum + B_r;
    float *prev_sum = prev_max + B_r;

    for(int j = 0 ; j < T_c; j++){

        //loading the K and V tiles
        int col = x % B_r;
        int row = x / B_r;
        for (int c = col; c < d ; c += B_r){ //coalescing
            k[row*d + c] = K[batch_idx * (h*s*d) + head_idx * (s*d) + (B_c *j*d + row*d)  + c];
            v[row*d + c] = V[batch_idx * (h*s*d) + head_idx * (s*d) + (B_c *j*d + row*d)  + c];
        }
        __syncthreads();
        
        for(int i = 0; i < T_r; i++){
            
            //loading the Q tiles
            int col = x % B_c;
            int row = x / B_c;

            for(int c = col ; c<d ; c += B_c){ //coalescing
                q[row*d + c]  = Q[batch_idx * (h*s*d) + head_idx * (s*d) + (B_r *i*d + row*d) + c];
            }
            __syncthreads();
          
            //computing the dot product
            float S_ij = 0.0f;
            for(int common = 0 ; common < d; common++){
                S_ij +=  q[row*d + common] * k[col*d + common];
            }
            S[row * B_c + col] = scale * S_ij;

            //computing the local max and local sum
            if(col==0){
              float local_mij = -INFINITY;
              float local_lij = 0.0f;
              for (int common = 0; common <B_c; common++){
                float curr_value = S[row * B_c + common];
                if(curr_value > local_mij){
                  local_lij = local_lij * expf(local_mij - curr_value);
                  local_mij = curr_value;
                }
                local_lij += expf(curr_value - local_mij);
              }
            local_max[row] = local_mij;
            local_sum[row] = local_lij;

            prev_max[row] = m[batch_idx * (h*s) + head_idx * s + B_r * i + row];
            prev_sum[row] = l[batch_idx * (h*s) + head_idx * s + B_r * i + row]; 
            row_max[row] = max(prev_max[row],local_max[row]);
            row_sum[row] = expf(prev_max[row]- row_max[row]) * prev_sum[row] + expf(local_max[row] - row_max[row]) * local_sum[row];
            }
            __syncthreads();

            
            //computing the final output
            for(int c = col; c < d; c += B_c){
              float output_sum = 0.0f;
              for(int common = 0; common < B_c; common++){

                output_sum += expf(S[row * B_c + common] - row_max[row]) * v[common*d+ c];
              }
              int idx = batch_idx * (h*s*d) + head_idx * (s*d) + (B_r *i*d + row*d) + c;
              O[idx] = output_sum /row_sum[row] + (O[idx] * expf(prev_max[row] - row_max[row]) * prev_sum[row]) / row_sum[row];
            }
            
            if(col==0){
              l[batch_idx * (h*s) + head_idx * s + B_r * i + row] = row_sum[row];
              m[batch_idx * (h*s) + head_idx * s + B_r * i + row] = row_max[row];
            }
            __syncthreads();
        }  
    }
}

torch::Tensor fa_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
  const int Bc = 16;
  const int Br = 16;

  int B = Q.size(0);
  int nh = Q.size(1);
  int N = Q.size(2);
  int d = Q.size(3);

  int Tc = ceil((float)N / Bc);
  int Tr = ceil((float)N / Br);
  float scale = 1.0 / sqrt(d);

  auto O = torch::zeros_like(Q);
  auto l = torch::zeros({B,nh,N});
  auto m = torch::full({B,nh,N} ,-INFINITY);
  torch::Device device(torch::kCUDA);
  l = l.to(device);
  m = m.to(device);

  const int smem_size = (2 * Bc * d + Br * d + Br * Bc + 6*Br) * sizeof(float);
  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, smem_size);

  dim3 grid_size(B, nh);     
  dim3 block_size(Br*Bc);  

  flash_attn<<<grid_size, block_size, smem_size>>>(
    K.data_ptr<float>(), V.data_ptr<float>(), Q.data_ptr<float>(), O.data_ptr<float>(), l.data_ptr<float>(), m.data_ptr<float>(),
    Tr, Tc, B, nh, N, d, Bc, Br,scale);
  return O;
}

int main() {
    int b = 1; // Batch size
    int h = 1; // Number of heads
    int s = 8; // Sequence length
    int d = 8; // Embedding dimension
    int B_c = 4; // Block size for keys/values
    int B_r = 4; // Block size for queries
    int T_r = s / B_r;
    int T_c = s / B_c;

    float scale = 1/sqrtf(d);
    size_t size = b * h * s * d * sizeof(float);
    float *h_K = (float*)malloc(size);
    float *h_V = (float*)malloc(size);
    float *h_Q = (float*)malloc(size);
    float *h_O = (float*)malloc(size);
    float *h_l = (float*)malloc(s * sizeof(float));
    float *h_m = (float*)malloc(s * sizeof(float));

    for (int i = 0; i < b * h * s * d; i++) {
        h_K[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0;
        h_V[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0;
        h_Q[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0;
        h_O[i] = 0.0f;
    }
    for (int i = 0; i < s; i++) {
        h_l[i] = 1.0f;
        h_m[i] = -INFINITY;
    }

    float *d_K, *d_V, *d_Q, *d_O, *d_l, *d_m;
    CHECK_CUDA(cudaMalloc((void**)&d_K, size));
    CHECK_CUDA(cudaMalloc((void**)&d_V, size));
    CHECK_CUDA(cudaMalloc((void**)&d_Q, size));
    CHECK_CUDA(cudaMalloc((void**)&d_O, size));
    CHECK_CUDA(cudaMalloc((void**)&d_l, s * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_m, s * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_K, h_K, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_O, h_O, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_l, h_l, s * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_m, h_m, s * sizeof(float), cudaMemcpyHostToDevice));

    dim3 gridDim(b, h);
    dim3 blockDim(B_r, B_c);
    size_t sharedMemSize = (2 * B_c * d + B_r * d + B_r * B_c) * sizeof(float);

    flash_attn<<<gridDim, blockDim, sharedMemSize>>>(d_K, d_V, d_Q, d_O, d_l, d_m,
                                                      T_r, T_c, b, h, s, d, B_c, B_r,scale);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_O, d_O, size, cudaMemcpyDeviceToHost));

    printf("Output tensor O (partial view):\n");
    for (int i = 0; i < s; i++) {
        for (int j = 0; j < d; j++) {
            printf("%.3f ", h_O[i * d + j]);
        }
        printf("\n");
    }

    free(h_K);
    free(h_V);
    free(h_Q);
    free(h_O);
    free(h_l);
    free(h_m);
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_O));
    CHECK_CUDA(cudaFree(d_l));
    CHECK_CUDA(cudaFree(d_m));

    return 0;
}


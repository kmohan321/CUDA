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



    __global__ void flash_attn(float *K, float *V, float *Q, float *O, float *l, float *m,
        int T_r, int T_c, int b, int h, int s, int d, int B_c, int B_r, float scale) {
     
        extern __shared__ float smem[];
    
        float *k = smem;
        float *v = k + B_c * d;
        float *q = v + B_c * d;
        float *S = q + B_r * d;
    
        int batch_idx = blockIdx.x;
        int head_idx = blockIdx.y;
        int x = threadIdx.x;
        
        int base_out_idx = batch_idx * (h*s*d) + head_idx * (s*d);
        
        for(int j = 0; j < T_c; j++) {
            for (int c = 0; c < d; c++) {
                k[x*d + c] = K[batch_idx * (h*s*d) + head_idx * (s*d) + (B_c*j*d + x*d) + c];
                v[x*d + c] = V[batch_idx * (h*s*d) + head_idx * (s*d) + (B_c*j*d + x*d) + c];
            }
            __syncthreads();
    
            for(int i = 0; i < T_r; i++) {
                for(int c = 0; c < d; c++) {
                    q[x*d + c] = Q[batch_idx * (h*s*d) + head_idx * (s*d) + (B_r*i*d + x*d) + c];
                }
                __syncthreads();
                
                float l_i = l[batch_idx * (h*s) + head_idx * s + B_r * i + x];
                float m_i = m[batch_idx * (h*s) + head_idx * s + B_r * i + x];
    
                float local_max = -INFINITY;
                for(int l_row = 0; l_row < B_c; l_row++) {
                    float S_ij = 0.0f;
                    for(int d_idx = 0; d_idx < d; d_idx++) {
                        S_ij += q[x*d + d_idx] * k[l_row*d + d_idx];
                    }
                    S[x * B_c + l_row] = S_ij * scale;
                    local_max = fmaxf(local_max, S[x * B_c + l_row]);
                }
                __syncthreads();
    
                float sum_exp = 0.0f;
                for(int l_row = 0; l_row < B_c; l_row++) {
                    sum_exp += expf(S[x * B_c + l_row] - local_max);
                }
    
                float m_new = fmaxf(m_i, local_max);
                float l_new = expf(m_i - m_new) * l_i + expf(local_max - m_new) * sum_exp;
                
                for(int d_idx = 0; d_idx < d; d_idx++) {
                    float out_val = 0.0f;
                    
                    for(int l_row = 0; l_row < B_c; l_row++) {
                        float att_weight = expf(S[x * B_c + l_row] - m_new);
                        out_val += att_weight * v[l_row*d + d_idx];
                    }
                    
                    int out_idx = base_out_idx + (B_r*i*d + x*d) + d_idx;
                    if (j == 0) {
                        O[out_idx] = out_val / l_new;
                    } else {
                        O[out_idx] = O[out_idx] * expf(m_i - m_new) * l_i / l_new + 
                                    out_val / l_new;
                    }
                }
    
                l[batch_idx * (h*s) + head_idx * s + B_r * i + x] = l_new;
                m[batch_idx * (h*s) + head_idx * s + B_r * i + x] = m_new;
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

  const int smem_size = (2 * Bc * d + Br * d + Br * Bc) * sizeof(float);
  int max_sram_size;
  cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, smem_size);

  dim3 grid_size(B, nh);     
  dim3 block_size(Br);  


  flash_attn<<<grid_size, block_size, smem_size>>>(
    K.data_ptr<float>(), V.data_ptr<float>(), Q.data_ptr<float>(), O.data_ptr<float>(), l.data_ptr<float>(), m.data_ptr<float>(),
    Tr, Tc, B, nh, N, d, Bc, Br,scale);
  return O;
}
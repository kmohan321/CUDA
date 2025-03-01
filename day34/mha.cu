#include "stdio.h"
#include <cuda_runtime.h>

__global__ void flash_attn(float *K, float *V, float *Q,float *O, float *l, float *m,
    int T_r, int T_c,int h, int s, int d,int B_c, int B_r, float scale){
 
    extern __shared__ float smem[];

    float *k = smem;
    float *v = k + B_c * d;
    float *q = v +  B_c * d;
    float *S = q + B_r * d;

    // int batch_idx = blockIdx.x;
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
            k[row*d + c] = K[head_idx * (s*d) + (B_c *j*d + row*d)  + c];
            v[row*d + c] = V[head_idx * (s*d) + (B_c *j*d + row*d)  + c];
        }
        __syncthreads();
        
        for(int i = 0; i < T_r; i++){
            
            //loading the Q tiles
            int col = x % B_c;
            int row = x / B_c;

            for(int c = col ; c<d ; c += B_c){ //coalescing
                q[row*d + c]  = Q[head_idx * (s*d) + (B_r *i*d + row*d) + c];
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
                  local_mij = curr_value;
                }
              }
              local_max[row] = local_mij;

              for(int common =0; common <B_c;common++){
                float curr_value = S[row * B_c + common];
                S[row * B_c + common] = expf(curr_value - local_max[row]);
                local_lij += S[row * B_c + common];
              }
              local_sum[row] = local_lij;

              prev_max[row] = m[head_idx * s + B_r * i + row];
              prev_sum[row] = l[head_idx * s + B_r * i + row]; 
              row_max[row] = max(prev_max[row],local_max[row]);
              row_sum[row] = expf(prev_max[row]- row_max[row]) * prev_sum[row] + expf(local_max[row] - row_max[row]) * local_sum[row];
            }
            __syncthreads();

            //computing the final output
            for(int c = col; c < d; c += B_c){
              float output_sum = 0.0f;
              for(int common = 0; common < B_c; common++){
                output_sum += S[row * B_c + common] * v[common*d+ c];
              }
              int idx = head_idx * (s*d) + (B_r *i*d + row*d) + c;
              O[idx] = output_sum * expf(local_max[row] - row_max[row]) /row_sum[row] + (O[idx] * expf(prev_max[row] - row_max[row]) * prev_sum[row]) / row_sum[row];
            }
            
            if(col==0){
              l[head_idx * s + B_r * i + row] = row_sum[row];
              m[head_idx * s + B_r * i + row] = row_max[row];
            }
        }
        __syncthreads();  
    }
}
__global__ void reshape_qkv(const float* input, float* output, int seq_len, int d_model, int num_heads, int d_k) {
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = blockIdx.y;
    
    if (seq_idx < seq_len) {
        for (int i = 0; i < d_k; i++) {

            int src_idx = seq_idx * d_model + head_idx * d_k + i;
            
            int dst_idx = head_idx * (seq_len * d_k) + seq_idx * d_k + i;
            
            output[dst_idx] = input[src_idx];
        }
    }
}

__global__ void reverse_reshape(const float* input, float* output, int seq_len, int d_model, int num_heads, int d_k) {
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_idx = blockIdx.y;
    
    if (seq_idx < seq_len) {
        for (int i = 0; i < d_k; i++) {
          
            int src_idx = head_idx * (seq_len * d_k) + seq_idx * d_k + i;
            
            int dst_idx = seq_idx * d_model + head_idx * d_k + i;
            
            output[dst_idx] = input[src_idx];
        }
    }
}

void solve(const float* Q, const float* K, const float* V, float* output, int N, int d_model, int h) {
   
    int d_k = d_model / h;   
    
    float *d_Q_orig, *d_K_orig, *d_V_orig, *d_O_orig;
    float *d_Q, *d_K, *d_V, *d_O;
    float *d_l, *d_m;  
    
    size_t orig_size = sizeof(float) * N * d_model;
    
    size_t reshaped_size = sizeof(float) * h * N * d_k;
    
    size_t tracking_size = sizeof(float) * h * N;

    cudaMalloc(&d_Q_orig, orig_size);
    cudaMalloc(&d_K_orig, orig_size);
    cudaMalloc(&d_V_orig, orig_size);
    cudaMalloc(&d_O_orig, orig_size);
    
    cudaMalloc(&d_Q, reshaped_size);
    cudaMalloc(&d_K, reshaped_size);
    cudaMalloc(&d_V, reshaped_size);
    cudaMalloc(&d_O, reshaped_size);
    
    cudaMalloc(&d_l, tracking_size);
    cudaMalloc(&d_m, tracking_size);
    
    cudaMemcpy(d_Q_orig, Q, orig_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K_orig, K, orig_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_orig, V, orig_size, cudaMemcpyHostToDevice);
    
    cudaMemset(d_O, 0, reshaped_size);
    
    float neg_inf = -INFINITY;
    float* h_m = (float*)malloc(tracking_size);
    float* h_l = (float*)malloc(tracking_size);
    
    for (int i = 0; i < h * N; i++) {
        h_m[i] = neg_inf;
        h_l[i] = 0.0f;
    }
    
    cudaMemcpy(d_m, h_m, tracking_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, h_l, tracking_size, cudaMemcpyHostToDevice);
    
    free(h_m);
    free(h_l);
    
    dim3 reshape_grid((N + 255) / 256, h);
    dim3 reshape_block(256);
    
    reshape_qkv<<<reshape_grid, reshape_block>>>(d_Q_orig, d_Q, N, d_model, h, d_k);
    reshape_qkv<<<reshape_grid, reshape_block>>>(d_K_orig, d_K, N, d_model, h, d_k);
    reshape_qkv<<<reshape_grid, reshape_block>>>(d_V_orig, d_V, N, d_model, h, d_k);
    
    int B_r = 32;  
    int B_c = 32;  
    
    int T_r = (N + B_r - 1) / B_r;  
    int T_c = (N + B_c - 1) / B_c;
    
    float scale = 1.0f / sqrtf(d_k);
    
    dim3 grid(1, h);  
    dim3 block(B_r * B_c);  
  
    size_t smem_size = sizeof(float) * (
        B_c * d_k +    // K tile
        B_c * d_k +    // V tile
        B_r * d_k +    // Q tile
        B_r * B_c +    // S (attention scores)
        B_r +          // local_max
        B_r +          // local_sum
        B_r +          // row_max
        B_r +          // row_sum
        B_r +          // prev_max
        B_r            // prev_sum
    );
    
    flash_attn<<<grid, block, smem_size>>>(
        d_K, d_V, d_Q, d_O, d_l, d_m,
        T_r, T_c, h, N, d_k, B_c, B_r, scale
    );
    
    reverse_reshape<<<reshape_grid, reshape_block>>>(d_O, d_O_orig, N, d_model, h, d_k);

    cudaMemcpy(output, d_O_orig, orig_size, cudaMemcpyDeviceToHost);
    cudaFree(d_Q_orig);
    cudaFree(d_K_orig);
    cudaFree(d_V_orig);
    cudaFree(d_O_orig);
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
    cudaFree(d_l);
    cudaFree(d_m);
}

// int main() {
//     const float q[2][4] = {{1.0, 0.0, 2.0, 3.0}, {4.0, 5.0, 6.0, 7.0}};
//     const float k[2][4] = {{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}};
//     const float v[2][4] = {{0.5, 1.0, 1.5, 2.0}, {2.5, 3.0, 3.5, 4.0}};
    
//     float output[2][4] = {0};

//     solve(&q[0][0], &k[0][0], &v[0][0], &output[0][0], 2, 4, 2);

//     printf("Output matrix:\n");
//     for (int i = 0; i < 2; i++) {
//         for (int j = 0; j < 4; j++) {
//             printf("%f ", output[i][j]);
//         }
//         printf("\n");
//     }

//     return 0;
// }

#include <cuda.h>
#include <iostream>

#define EPSILON 1e-5
#define BLOCK_SIZE 1024  

__global__ void layernorm_relu(float* x, float* y, float* gamma, float* beta, int C) {
    extern __shared__ float shared_mem[];  
    float* s_mean = shared_mem;            
    float* s_var = &shared_mem[blockDim.x]; 

    int batch_idx = blockIdx.x;
    int feature_idx = threadIdx.x;

    x += batch_idx * C;
    y += batch_idx * C;

    float val = (feature_idx < C) ? x[feature_idx] : 0.0f;
    float sum = val;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (feature_idx < stride)
            sum += s_mean[feature_idx + stride];
        s_mean[feature_idx] = sum;
    }
    __syncthreads();

    float mean = s_mean[0] / C;

    float diff = (feature_idx < C) ? (val - mean) * (val - mean) : 0.0f;
    float var_sum = diff;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (feature_idx < stride)
            var_sum += s_var[feature_idx + stride];
        s_var[feature_idx] = var_sum;
    }
    __syncthreads();

    float variance = s_var[0] / C;
    float std_inv = rsqrtf(variance + EPSILON);

    if (feature_idx < C) {
        float norm_x = (x[feature_idx] - mean) * std_inv;
        float normalized = gamma[feature_idx] * norm_x + beta[feature_idx];

        y[feature_idx] = fmaxf(0.0f, normalized);
    }
}

void run_layernorm_relu(float* h_x, float* h_y, float* h_gamma, float* h_beta, int N, int C) {
    float *d_x, *d_y, *d_gamma, *d_beta;
    
    size_t size = N * C * sizeof(float);
    
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMalloc(&d_gamma, C * sizeof(float));
    cudaMalloc(&d_beta, C * sizeof(float));

    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, C * sizeof(float), cudaMemcpyHostToDevice);

    int shared_mem_size = 2 * BLOCK_SIZE * sizeof(float); 
    layernorm_relu<<<N, BLOCK_SIZE, shared_mem_size>>>(d_x, d_y, d_gamma, d_beta, C);

    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_gamma);
    cudaFree(d_beta);
}

int main() {
    int N = 2, C = 4;  
    float h_x[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float h_y[8];
    float h_gamma[] = {1.0, 1.0, 1.0, 1.0};  
    float h_beta[] = {0.0, 0.0, 0.0, 0.0};  

    run_layernorm_relu(h_x, h_y, h_gamma, h_beta, N, C);

    std::cout << "Output after LayerNorm + ReLU: ";
    for (int i = 0; i < N * C; i++) {
        std::cout << h_y[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}


#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define BATCH_SIZE 128
#define FEATURE_DIM 65536 //huge number right?
#define BLOCK_SIZE 1024 

void layernorm_cpu(float *input, float *output, float *gamma, float *beta) {
    for (int batch = 0; batch < BATCH_SIZE; batch++) {
        float mean = 0.0f, var = 0.0f;
        
        // Compute mean
        for (int i = 0; i < FEATURE_DIM; i++) {
            mean += input[batch * FEATURE_DIM + i];
        }
        mean /= FEATURE_DIM;
        
        // Compute variance
        for (int i = 0; i < FEATURE_DIM; i++) {
            float diff = input[batch * FEATURE_DIM + i] - mean;
            var += diff * diff;
        }
        var /= FEATURE_DIM;
        float inv_std = 1.0f / sqrtf(var + 1e-5f);
        
        // Normalize, scale, and shift
        for (int i = 0; i < FEATURE_DIM; i++) {
            int idx = batch * FEATURE_DIM + i;
            output[idx] = gamma[i] * ((input[idx] - mean) * inv_std) + beta[i];
        }
    }
}
/*each block processes one row*/
__global__ void layernorm_kernel(float *input, float *output, float *gamma, float *beta) {
    __shared__ float s_mean[BLOCK_SIZE];
    __shared__ float s_var[BLOCK_SIZE];

    int x = threadIdx.x;
    int row = blockIdx.x;  
    float sum = 0.0f, temp = 0.0f;

    // local sum
    for (int stride = x; stride < FEATURE_DIM; stride += BLOCK_SIZE) {
        sum += input[stride + row * FEATURE_DIM];
    }
    
    s_mean[x] = sum;
    __syncthreads();

    // reduction time
    for (int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
        if (x < i) {
            s_mean[x] += s_mean[x + i];
        }
        __syncthreads();
    }

    float mean = s_mean[0] / FEATURE_DIM;

    // variance time
    for (int stride = x; stride < FEATURE_DIM; stride += BLOCK_SIZE) {
        float diff = input[stride + row * FEATURE_DIM] - mean;
        temp += diff * diff;
    }

    s_var[x] = temp;
    __syncthreads();

    // reduction time
    for (int i = BLOCK_SIZE / 2; i > 0; i /= 2) {
        if (x < i) {
            s_var[x] += s_var[x + i];
        }
        __syncthreads();
    }

    float variance = s_var[0] / FEATURE_DIM;
    float inv_std = 1.0f / sqrtf(variance + 1e-5f);

    // Normalizing 
    for (int stride = x; stride < FEATURE_DIM; stride += BLOCK_SIZE) {
        output[stride + row * FEATURE_DIM] = 
            ((input[stride + row * FEATURE_DIM] - mean) * inv_std) * gamma[stride] + beta[stride];
    }
}

int main() {
    float *h_input, *h_output, *h_gamma, *h_beta, *h_output_cpu;
    float *d_input, *d_output, *d_gamma, *d_beta;
    
    size_t size = BATCH_SIZE * FEATURE_DIM * sizeof(float);
    h_input = (float*)malloc(size);
    h_output = (float*)malloc(size);
    h_output_cpu = (float*)malloc(size);
    h_gamma = (float*)malloc(FEATURE_DIM * sizeof(float));
    h_beta = (float*)malloc(FEATURE_DIM * sizeof(float));

    for (int i = 0; i < BATCH_SIZE * FEATURE_DIM; i++) {
        h_input[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < FEATURE_DIM; i++) {
        h_gamma[i] = 1.0f;
        h_beta[i] = 0.0f;
    }

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_gamma, FEATURE_DIM * sizeof(float));
    cudaMalloc(&d_beta, FEATURE_DIM * sizeof(float));

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, FEATURE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, FEATURE_DIM * sizeof(float), cudaMemcpyHostToDevice);

    clock_t start_cpu = clock();
    layernorm_cpu(h_input, h_output_cpu, h_gamma, h_beta);
    clock_t end_cpu = clock();
    double cpu_time = 1000.0 * (end_cpu - start_cpu) / CLOCKS_PER_SEC;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 grid(BATCH_SIZE);
    dim3 block(BLOCK_SIZE);

    cudaEventRecord(start);
    layernorm_kernel<<<grid, block>>>(d_input, d_output, d_gamma, d_beta);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    float max_error = 0.0f;
    for (int i = 0; i < BATCH_SIZE * FEATURE_DIM; i++) {
        max_error = fmaxf(max_error, fabs(h_output[i] - h_output_cpu[i]));
    }

    printf("GPU Time (CUDA Event Timing): %f ms\n", milliseconds);
    printf("CPU Time: %f ms\n", cpu_time);
    printf("Max error: %f\n", max_error);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(h_input);
    free(h_output);
    free(h_output_cpu);
    free(h_gamma);
    free(h_beta);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_gamma);
    cudaFree(d_beta);

    return 0;
}

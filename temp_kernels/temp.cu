#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>

#define r 1000 // rows
#define c 1000 // columns
#define k 500// common dimension
#define blocksize 32

void cpu_matrix_mult(float *A, float *B, float *C) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * c + j];
            }
            C[i * c + j] = sum;
        }
    }
}

__global__ void my_kernel(float *A, float *output, float *B) {
    __shared__ float s1[blocksize * blocksize];
    __shared__ float s2[blocksize * blocksize];

    int x = threadIdx.x % blocksize; // column index in tile
    int y = threadIdx.x / blocksize; // row index in tile

    int row = blockIdx.y * blocksize + y;
    int col = blockIdx.x * blocksize + x;

    float sum = 0.0f;
    for (int tileid = 0; tileid < (k + blocksize - 1) / blocksize; tileid++) {
        if (row < r && (tileid * blocksize + x) < k) {
            s1[x + y * blocksize] = A[row * k + tileid * blocksize + x];
        } else {
            s1[x + y * blocksize] = 0.0f;
        }

        if (col < c && (tileid * blocksize + y) < k) {
            s2[x + y * blocksize] = B[(tileid * blocksize + y) * c + col];
        } else {
            s2[x + y * blocksize] = 0.0f;
        }
        __syncthreads();

        for (int i = 0; i < blocksize; i++) {
            sum += s1[i + y * blocksize] * s2[x + i * blocksize];
        }
        __syncthreads();
    }

    if (row < r && col < c) {
        output[row * c + col] = sum;
    }
}

int main() {
    srand(time(NULL));

    float *h_A = (float *)malloc(r * k * sizeof(float));
    float *h_B = (float *)malloc(k * c * sizeof(float));
    float *h_C = (float *)malloc(r * c * sizeof(float));
    float *h_C_gpu = (float *)malloc(r * c * sizeof(float));

    // Randomly initialize matrices A and B
    for (int i = 0; i < r * k; i++) {
        h_A[i] = (float)(rand() % 10);
    }
    for (int i = 0; i < k * c; i++) {
        h_B[i] = (float)(rand() % 10);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, r * k * sizeof(float));
    cudaMalloc((void **)&d_B, k * c * sizeof(float));
    cudaMalloc((void **)&d_C, r * c * sizeof(float));

    cudaMemcpy(d_A, h_A, r * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * c * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(blocksize * blocksize);
    dim3 gridDim((c + blocksize - 1) / blocksize, (r + blocksize - 1) / blocksize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    my_kernel<<<gridDim, blockDim>>>(d_A, d_C, d_B);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(h_C_gpu, d_C, r * c * sizeof(float), cudaMemcpyDeviceToHost);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);
    printf("GPU Time: %f ms\n", gpu_time);

    // CPU computation timing
    clock_t cpu_start = clock();
    cpu_matrix_mult(h_A, h_B, h_C);
    clock_t cpu_end = clock();
    double cpu_time = 1000.0 * (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("CPU Time: %f ms\n", cpu_time);

    // Compare GPU results with CPU results
    float max_error = 0.0f;
    for (int i = 0; i < r * c; i++) {
        max_error = fmax(max_error, fabs(h_C[i] - h_C_gpu[i]));
    }

    printf("Max error: %f\n", max_error);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_gpu);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

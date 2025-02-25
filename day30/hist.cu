#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16  
#define HISTOGRAM_SIZE 64 

__global__ void compute_histogram(unsigned char *d_img, unsigned int *d_hist, int width, int height) {
    __shared__ unsigned int local_hist[HISTOGRAM_SIZE]; 
    
    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;

    int index = y * width + x;
    
    if (tx < HISTOGRAM_SIZE / BLOCK_SIZE && ty == 0) {
        local_hist[tx * BLOCK_SIZE] = 0;
    }
    __syncthreads();

    if (x < width && y < height) {
        atomicAdd(&local_hist[d_img[index]], 1);
    }
    __syncthreads();

    if (tx == 0 && ty == 0) {
        for (int i = 0; i < HISTOGRAM_SIZE; i++) {
            atomicAdd(&d_hist[i], local_hist[i]);
        }
    }
}

__global__ void apply_histogram_equalization(unsigned char *d_img, unsigned char *d_out, unsigned int *d_hist, int width, int height) {
    __shared__ float cdf[HISTOGRAM_SIZE];    
    
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float sum = 0;
        for (int i = 0; i < HISTOGRAM_SIZE; i++) {
            sum += d_hist[i];
            cdf[i] = sum;
        }

        float min_cdf = cdf[0];
        for (int i = 0; i < HISTOGRAM_SIZE; i++) {
            cdf[i] = ((cdf[i] - min_cdf) / (width * height - min_cdf)) * 255.0f;
        }
    }
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = y * width + x;

    if (x < width && y < height) {
        d_out[index] = (unsigned char)cdf[d_img[index]];
    }
}

int main() {
    int width = 512, height = 512;
    size_t img_size = width * height * sizeof(unsigned char);
    size_t hist_size = HISTOGRAM_SIZE * sizeof(unsigned int);

    unsigned char *h_img = (unsigned char*)malloc(img_size);
    unsigned char *h_out = (unsigned char*)malloc(img_size);
    unsigned int *h_hist = (unsigned int*)malloc(hist_size);

    for (int i = 0; i < width * height; i++) {
        h_img[i] = rand() % HISTOGRAM_SIZE;
    }

    unsigned char *d_img, *d_out;
    unsigned int *d_hist;

    cudaMalloc(&d_img, img_size);
    cudaMalloc(&d_out, img_size);
    cudaMalloc(&d_hist, hist_size);
    cudaMemset(d_hist, 0, hist_size);

    cudaMemcpy(d_img, h_img, img_size, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    compute_histogram<<<gridDim, blockDim>>>(d_img, d_hist, width, height);
    cudaDeviceSynchronize();

    apply_histogram_equalization<<<gridDim, blockDim>>>(d_img, d_out, d_hist, width, height);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, img_size, cudaMemcpyDeviceToHost);

    printf("Histogram equalization completed.\n");

    cudaFree(d_img);
    cudaFree(d_out);
    cudaFree(d_hist);
    free(h_img);
    free(h_out);
    free(h_hist);

    return 0;
}

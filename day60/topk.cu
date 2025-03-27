#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <vector>
#include <algorithm>


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s (err_num=%d)\n", cudaGetErrorString(err), err); \
            exit(err); \
        } \
    } while (0)

__device__ void heapify_down(float* heap, int k, int idx) {
    int smallest = idx;
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;

    if (left < k && heap[left] < heap[smallest]) 
        smallest = left;
    
    if (right < k && heap[right] < heap[smallest]) 
        smallest = right;
    
    if (smallest != idx) {
        float temp = heap[idx];
        heap[idx] = heap[smallest];
        heap[smallest] = temp;
        
        
        heapify_down(heap, k, smallest);
    }
}


__device__ void heap_insert(float* heap, int k, float value) {
    if (value > heap[0]) {
        heap[0] = value;
        heapify_down(heap, k, 0);
    }
}

__global__ void top_k_per_block(float* data, float* top_k_results, int N, int k) {
  
    int chunkSize = N / gridDim.x;  
    int start = blockIdx.x * chunkSize;
    int end = min(start + chunkSize, N);

    __shared__ float heap[1024]; 

    if (threadIdx.x < k) {
        heap[threadIdx.x] = (start + threadIdx.x < end) ? data[start + threadIdx.x] : -INFINITY;
    }
    __syncthreads();

    for (int i = start + k + threadIdx.x; i < end; i += blockDim.x) {
        heap_insert(heap, k, data[i]);
    }
    __syncthreads();

    if (threadIdx.x < k) {
        top_k_results[blockIdx.x * k + threadIdx.x] = heap[threadIdx.x];
    }
}

void merge_top_k(float* data, int total_k, int final_k) {
    std::make_heap(data, data + total_k, std::greater<float>()); // Min-heap
    std::sort_heap(data, data + total_k, std::greater<float>()); // Get top-K
}

int main() {
    int N = 1 << 20; 
    int k = 10;      
    int num_blocks = 128;
    int threads_per_block = 256;

    float *h_data = (float*)malloc(N * sizeof(float));
    float *h_top_k = (float*)malloc(num_blocks * k * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(rand() % 10000);
    }

    float *d_data, *d_top_k;
    CUDA_CHECK(cudaMalloc((void**)&d_data, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_top_k, num_blocks * k * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

    top_k_per_block<<<num_blocks, threads_per_block>>>(d_data, d_top_k, N, k);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_top_k, d_top_k, num_blocks * k * sizeof(float), cudaMemcpyDeviceToHost));

    merge_top_k(h_top_k, num_blocks * k, k);

    printf("Top-%d elements:\n", k);
    for (int i = 0; i < k; i++) {
        printf("%f\n", h_top_k[i]);
    }

    free(h_data);
    free(h_top_k);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_top_k));

    return 0;
}

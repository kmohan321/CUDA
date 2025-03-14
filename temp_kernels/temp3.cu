
#include <cuda.h>
#include <stdio.h>

#define blocksize 32
#define k 500
#define r 1000
#define c 1000

__global__ void dx(float* dx, float* dy, float* W_t) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    __shared__ float s1[blocksize * blocksize];
    __shared__ float s2[blocksize * blocksize];

    int x = threadIdx.x;
    int y = threadIdx.y;
    float sum = 0.0f;

    for (int tileid = 0; tileid < ceil((float)k / blocksize); tileid++) {
        if (row < r && (x + tileid * blocksize) < k) {
            s1[x + y * blocksize] = dy[x + row * k + tileid * blocksize];
        }
        else{
            s1[x + y * blocksize] = 0.0f;
        }

        if ((y + blocksize * tileid) < k && col < c) {
            s2[x + y * blocksize] = W_t[col + y * c + c * (tileid * blocksize)];
        }
        else{
            s2[x + y * blocksize]= 0.0f;
        }
        __syncthreads();

        for (int sid = 0; sid < blocksize; sid++) {
            sum += s1[sid + blocksize * y] * s2[x + sid * blocksize];
        } //maybe outofbond memory access for shared memory
      __syncthreads();

    }
    if (row < r && col < c) {
        dx[col + row * c] = sum;
    }
}

// CPU implementation for validation
void cpu_matrix_multiply(float* dx, float* dy, float* W_t) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            float sum = 0.0f;
            for (int m = 0; m < k; m++) {
                sum += dy[m + i * k] * W_t[j + m * c];
            }
            dx[j + i * c] = sum;
        }
    }
}

// Utility function to check results
bool compare_results(float* cpu_result, float* gpu_result, int size) {
    const float epsilon = 1e-2;  // Increased tolerance
    const float rel_epsilon = 1e-3;  // Relative error tolerance
    int mismatches = 0;
    float max_diff = 0.0f;
    
    for (int i = 0; i < size; i++) {
        float abs_diff = fabsf(cpu_result[i] - gpu_result[i]);
        float rel_diff = abs_diff / (fabsf(cpu_result[i]) + 1e-6);
        
        if (abs_diff > epsilon && rel_diff > rel_epsilon) {
            if (mismatches < 5) {  // Print first 5 mismatches
                printf("Mismatch at index %d: CPU = %f, GPU = %f (diff = %f, rel_diff = %f)\n",
                       i, cpu_result[i], gpu_result[i], abs_diff, rel_diff);
            }
            max_diff = fmaxf(max_diff, abs_diff);
            mismatches++;
        }
    }
    
    if (mismatches > 0) {
        printf("Total mismatches: %d\n", mismatches);
        printf("Maximum absolute difference: %f\n", max_diff);
        return false;
    }
    return true;
}

int main() {
    // Allocate host memory
    float *h_dx, *h_dy, *h_W_t;
    float *h_dx_cpu;  // For CPU results
    
    h_dx = (float*)malloc(r * c * sizeof(float));
    h_dy = (float*)malloc(r * k * sizeof(float));
    h_W_t = (float*)malloc(k * c * sizeof(float));
    h_dx_cpu = (float*)malloc(r * c * sizeof(float));

    // Initialize input matrices with random values
    srand(time(NULL));
    for (int i = 0; i < r * k; i++) {
        h_dy[i] = (float)(rand()) / RAND_MAX;
    }
    for (int i = 0; i < k * c; i++) {
        h_W_t[i] = (float)(rand()) / RAND_MAX;
    }

    // Allocate device memory
    float *d_dx, *d_dy, *d_W_t;
    cudaMalloc(&d_dx, r * c * sizeof(float));
    cudaMalloc(&d_dy, r * k * sizeof(float));
    cudaMalloc(&d_W_t, k * c * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_dy, h_dy, r * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_t, h_W_t, k * c * sizeof(float), cudaMemcpyHostToDevice);

    // Set up timing for GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Configure kernel launch parameters
    dim3 threadsPerBlock(blocksize, blocksize);
    dim3 numBlocks((c + blocksize - 1) / blocksize, 
                   (r + blocksize - 1) / blocksize);

    // Launch kernel and measure time
    cudaEventRecord(start);
    dx<<<numBlocks, threadsPerBlock>>>(d_dx, d_dy, d_W_t);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate GPU time
    float gpu_milliseconds = 0;
    cudaEventElapsedTime(&gpu_milliseconds, start, stop);

    // Copy result back to host
    cudaMemcpy(h_dx, d_dx, r * c * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU implementation timing
    clock_t cpu_start = clock();
    cpu_matrix_multiply(h_dx_cpu, h_dy, h_W_t);
    clock_t cpu_end = clock();
    double cpu_milliseconds = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // Compare results
    bool results_match = compare_results(h_dx_cpu, h_dx, r * c);

    // Print timing results
    printf("GPU Time: %f ms\n", gpu_milliseconds);
    printf("CPU Time: %f ms\n", cpu_milliseconds);
    printf("Speedup: %.2fx\n", cpu_milliseconds / gpu_milliseconds);
    printf("Results match: %s\n", results_match ? "Yes" : "No");

    // Clean up
    cudaFree(d_dx);
    cudaFree(d_dy);
    cudaFree(d_W_t);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(h_dx);
    free(h_dy);
    free(h_W_t);
    free(h_dx_cpu);

    return 0;
}
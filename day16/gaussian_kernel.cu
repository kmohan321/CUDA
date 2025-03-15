#include <stdio.h>
#include <cuda.h>
#include <math.h>

#define BLOCK_SIZE 16
#define KERNEL_SIZE 5  // 5x5 Gaussian kernel
#define SIGMA 1.0f    // Blur strength
#define M_PI 3.14159265f

// You'll implement this
__global__ void gaussianBlur(unsigned char* input, unsigned char* output, 
                            float* kernel, int width, int height){
            
        int x = threadIdx.x;
        int y = threadIdx.y;

        int idx = x + blockDim.x * blockIdx.x;
        int idy = y + blockDim.y * blockIdx.y;
        
        const int shared_size = BLOCK_SIZE + KERNEL_SIZE -1;
        //reason -> (blocksize + kernel -1) => because of halo region
        __shared__ float smem[shared_size * shared_size];

        int radius = KERNEL_SIZE / 2;
        int shared_width = BLOCK_SIZE + 2 * radius;  
        int shared_height = BLOCK_SIZE + 2 * radius;

        if(idx<width && idy<height){

        }
        //not completed will work tomorrow

                            }

// Helper function to create Gaussian kernel
void createGaussianKernel(float* kernel, int kernelSize, float sigma) {
    float sum = 0.0f;
    int radius = kernelSize / 2;
    
    // Generate kernel values
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            float exponent = -(x*x + y*y)/(2*sigma*sigma);
            float value = exp(exponent)/(2*M_PI*sigma*sigma);
            kernel[(y + radius)*kernelSize + (x + radius)] = value;
            sum += value;
        }
    }
    
    // Normalize kernel
    for (int i = 0; i < kernelSize*kernelSize; i++) {
        kernel[i] /= sum;
    }
}

int main() {
    // Image dimensions
    int width = 1024;  // Example size
    int height = 1024;
    int imageSize = width * height * sizeof(unsigned char);
    
    // Allocate host memory
    unsigned char *h_input = (unsigned char*)malloc(imageSize);
    unsigned char *h_output = (unsigned char*)malloc(imageSize);
    float *h_kernel = (float*)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    
    // Create Gaussian kernel
    createGaussianKernel(h_kernel, KERNEL_SIZE, SIGMA);
    
    // Load your image into h_input here
    // For testing, you can generate a pattern:
    for(int i = 0; i < width*height; i++) {
        h_input[i] = i % 256;
    }
    
    // Allocate device memory
    unsigned char *d_input, *d_output;
    float *d_kernel;
    cudaMalloc(&d_input, imageSize);
    cudaMalloc(&d_output, imageSize);
    cudaMalloc(&d_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, imageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // Set up grid and block dimensions
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                  (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Launch kernel
    gaussianBlur<<<gridSize, blockSize>>>(d_input, d_output, d_kernel, width, height);
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost);
    
    // Verify results - print some sample pixels
    printf("Sample output pixels:\n");
    for(int i = 0; i < 5; i++) {
        printf("%d ", h_output[i]);
    }
    printf("\n");
    
    // Free memory
    free(h_input);
    free(h_output);
    free(h_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    
    return 0;
}
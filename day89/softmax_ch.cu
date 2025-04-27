#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#include <cuda_runtime.h>

__global__ void softmax(const float *__restrict__ input, float *__restrict__ output, int M, int N, int dim, size_t *shape, int ndim){

    int row = blockIdx.x; //will give the elements over the remaining dimension, need reformulation
    int idx = threadIdx.x; 

    int tmp = row;
    int idxs[5] = {0};
    for (int i = ndim - 1; i >= 0; --i) {
        if (i == dim) {
            idxs[i] = 0;
        } else {
            idxs[i] = tmp % shape[i];  
            tmp /= shape[i];
        }
    }

    int strides[5] = {0};
    strides[ndim-1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    extern __shared__ float smem[];
    float *smem1 = smem;
    float *smem2 = smem1 + blockDim.x;

    float local_max = -INFINITY;
    float local_sum = 0.0f;
    for(int stride = idx; stride < N; stride += blockDim.x){
        if(stride < N){
            idxs[dim] = stride;
            int offset = 0;
            for (int i = 0; i < ndim; ++i) {
                offset += idxs[i] * strides[i];
            }
            float curr_value = input[offset];
            if(curr_value > local_max){
                local_sum = local_sum * expf(local_max - curr_value);
                local_max = curr_value;
            }
            local_sum += expf(curr_value - local_max);
        }
    }

    if(idx<N){
        smem1[idx] = local_max;
    }
    else{
        smem1[idx] = -INFINITY;
    }
    
    __syncthreads();

    //reduction time
    for(int i = blockDim.x/2 ; i>0; i/=2){
        if(idx<i){
            smem1[idx] = fmaxf(smem1[idx] , smem1[idx+i]);
        }
        __syncthreads();
    }

    float global_max = smem1[0];
    if(idx<N){
        smem2[idx] = local_sum * expf(local_max - global_max);
    }
    else{
        smem2[idx] = 0.0f;
    }
    
    __syncthreads();

    for(int i = blockDim.x/2 ; i>0; i/=2){
        if(idx<i){
            smem2[idx] += smem2[idx + i];
        }
        __syncthreads();
    }

    float global_sum = smem2[0];
    for(int stride = idx; stride < N; stride += blockDim.x){
        if(stride < N){
            idxs[dim] = stride;
            int offset = 0;
            for (int i = 0; i < ndim; ++i) {
                offset += idxs[i] * strides[i];
            }
            float curr_value = input[offset];
            output[offset] = expf(curr_value - global_max) / global_sum;
        }   
    }

}


// Note: input, output, shape are all device pointers to float32 arrays
extern "C" void solution(const float* input, int dim, float* output, size_t* shape, size_t ndim) {    

    int threads = 512;
    size_t* h_shape = new size_t[ndim];
    cudaMemcpy(h_shape, shape, ndim * sizeof(size_t), cudaMemcpyDeviceToHost);
    int N = h_shape[dim];
    int M = 1;
    for(int i = 0; i < ndim; i++){
        if(i !=dim){
            M *= h_shape[i];
        }
    }
    dim3 blocksize(threads);
    dim3 grid(M);
    int smem_size = 2 * threads * sizeof(float);
    softmax<<<grid, blocksize, smem_size>>>(input, output, M,N,dim, shape, ndim);

}

// CPU implementation of softmax for validation
void cpu_softmax(const float* input, int dim, float* output, size_t* shape, size_t ndim) {
    // Calculate total size and stride information
    int total_size = 1;
    for (int i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    
    int dim_size = shape[dim];
    
    // Calculate strides
    int strides[5] = {0};
    strides[ndim-1] = 1;
    for (int i = ndim - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    
    // Process each batch of elements along the dimension
    int batch_size = total_size / dim_size;
    
    for (int batch = 0; batch < batch_size; batch++) {
        // Convert batch index to multi-dimensional indices
        int indices[5] = {0};
        int tmp = batch;
        for (int i = 0; i < ndim; i++) {
            if (i != dim) {
                int adjusted_dim = (i < dim) ? i : i - 1;
                indices[i] = tmp % shape[i];
                tmp /= shape[i];
            }
        }
        
        // Find max value for numerical stability
        float max_val = -INFINITY;
        for (int j = 0; j < dim_size; j++) {
            indices[dim] = j;
            int offset = 0;
            for (int i = 0; i < ndim; i++) {
                offset += indices[i] * strides[i];
            }
            max_val = fmax(max_val, input[offset]);
        }
        
        // Calculate sum of exp(x - max_val)
        float sum = 0.0f;
        for (int j = 0; j < dim_size; j++) {
            indices[dim] = j;
            int offset = 0;
            for (int i = 0; i < ndim; i++) {
                offset += indices[i] * strides[i];
            }
            sum += expf(input[offset] - max_val);
        }
        
        // Calculate softmax values
        for (int j = 0; j < dim_size; j++) {
            indices[dim] = j;
            int offset = 0;
            for (int i = 0; i < ndim; i++) {
                offset += indices[i] * strides[i];
            }
            output[offset] = expf(input[offset] - max_val) / sum;
        }
    }
}

// Utility function to check for CUDA errors
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Test function
void test_softmax(size_t* shape, size_t ndim, int dim) {
    // Calculate total size
    size_t total_size = 1;
    for (size_t i = 0; i < ndim; i++) {
        total_size *= shape[i];
    }
    
    // Allocate and initialize host memory
    float* h_input = (float*)malloc(total_size * sizeof(float));
    float* h_output = (float*)malloc(total_size * sizeof(float));
    float* h_expected = (float*)malloc(total_size * sizeof(float));
    
    // Initialize input with random values
    for (size_t i = 0; i < total_size; i++) {
        h_input[i] = (float)(rand() % 10000) / 1000.0f - 5.0f;  // Random values between -5 and 5
    }
    
    // Allocate device memory
    float* d_input;
    float* d_output;
    size_t* d_shape;
    
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_input, total_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_output, total_size * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_shape, ndim * sizeof(size_t)));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, total_size * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_shape, shape, ndim * sizeof(size_t), cudaMemcpyHostToDevice));
    
    // Call the solution function
    solution(d_input, dim, d_output, d_shape, ndim);
    
    // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, total_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Calculate expected results on CPU
    cpu_softmax(h_input, dim, h_expected, shape, ndim);
    
    // Verify results
    bool passed = true;
    float max_diff = 0.0f;
    for (size_t i = 0; i < total_size; i++) {
        float diff = fabsf(h_output[i] - h_expected[i]);
        max_diff = fmax(max_diff, diff);
        if (diff > 1e-5) {
            passed = false;
            printf("Mismatch at index %zu: GPU = %f, CPU = %f, diff = %f\n", 
                   i, h_output[i], h_expected[i], diff);
            if (i > 10) break;  // Limit the number of error messages
        }
    }
    
    if (passed) {
        printf("Test PASSED! Max difference: %e\n", max_diff);
    } else {
        printf("Test FAILED! Max difference: %e\n", max_diff);
    }
    
    // Print a few values for visual inspection
    printf("Sample results (first 10 elements):\n");
    for (int i = 0; i < 10 && i < total_size; i++) {
        printf("  [%d] Input: %f, GPU: %f, CPU: %f\n", i, h_input[i], h_output[i], h_expected[i]);
    }
    
    // Free memory
    free(h_input);
    free(h_output);
    free(h_expected);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_shape);
}

int main() {
    // Test case 1: 2D tensor, softmax along dim 1
    {
        size_t shape[4] = {4, 256, 256, 256};
        test_softmax(shape, 4, 3);
    }
    
    // Test case 2: 3D tensor, softmax along dim 0
    {
        size_t shape[2] = {128, 10};
        test_softmax(shape, 2, 1);
    }
    
    // Test case 3: 4D tensor, softmax along dim 2
    {
        size_t shape[3] = {256,50,50};
        test_softmax(shape, 3, 0);
    }
    
    // Test case 4: Small tensor for debugging
    {
        size_t shape[4] = {64, 128, 128, 128};
        test_softmax(shape, 4, 2);
    }
    
    // Test case 5: Large tensor to test performance
    {
        size_t shape[3] = {8, 1024, 1024};
        test_softmax(shape, 3, 1);
    }

    {
        size_t shape[3] = {32, 512, 512};
        test_softmax(shape, 3, 2);
    }
    {
        size_t shape[3] = {16, 128, 256};
        test_softmax(shape, 3, 1);
    }
    
    return 0;
}
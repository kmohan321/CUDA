#include <iostream>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>

#define N 1024
#define C 65536
#define blocksize 1024

__global__ void softmaxCrossEntropy(float* logits, int* labels, float* loss){

    int x = threadIdx.x;
    int row = blockIdx.x;
    if(row >= N) return;
    __shared__ float s[blocksize];
    float local_max = -INFINITY;
    float norm = 0.0f;

    //local sum
    for(int id = x ; id < C; id += blocksize){
        float curr_value = logits[id + row*C];
        if(curr_value>local_max){
            norm = norm * expf(local_max-curr_value);
            local_max = curr_value; 
        }
        norm += expf(curr_value-local_max);
    }
    __syncthreads();
    s[x] = local_max;
    __syncthreads();

    //reduction time
    for(int i = blockDim.x/2;i>0;i/=2){
        if(x<i){
            s[x]= fmax(s[x+i],s[x]);
        }
        __syncthreads();
    }

    float global_max = s[0];
    __syncthreads();
    norm = norm * expf(local_max-global_max);
    s[x] = norm;
    __syncthreads();

    //reduction time 
    for(int j =blockDim.x/2;j>0;j/=2){
        if(x<j){
            s[x] += s[x+j];
        }
        __syncthreads();
    }
    
    float final_norm = s[0];
    __syncthreads();
    
    int curr_label = labels[row];
    float softmax_value = expf(logits[curr_label + row*C] - global_max)/final_norm;
    loss[row] = -logf(softmax_value);

}

// GPU function prototype (You will implement this)
void softmaxCrossEntropyGPU(float* logits, int* labels, float* loss){
    
    dim3 gridsize (N);
    dim3 threadperblock (blocksize);

    softmaxCrossEntropy<<<gridsize, threadperblock>>>(logits,labels,loss);

}

// CPU Implementation of Softmax + Cross-Entropy
void softmaxCrossEntropyCPU(float* logits, int* labels, float* loss, int batch_size, int num_classes) {
    for (int i = 0; i < batch_size; i++) {
        // Compute softmax
        float max_logit = logits[i * num_classes];  // Prevent overflow
        for (int j = 1; j < num_classes; j++) 
            max_logit = fmax(max_logit, logits[i * num_classes + j]);
        
        float sum_exp = 0.0f;
        for (int j = 0; j < num_classes; j++) 
            sum_exp += expf(logits[i * num_classes + j] - max_logit);
        
        // Compute loss
        int label = labels[i];  // True class index
        float softmax_prob = expf(logits[i * num_classes + label] - max_logit) / sum_exp;
        loss[i] = -logf(softmax_prob);
    }
}

int main() {
    // Allocate memory using malloc
    float *h_logits = (float*) malloc(N * C * sizeof(float));  // Logits (random input)
    int *h_labels = (int*) malloc(N * sizeof(int));            // Labels (random integers)
    float *h_loss_cpu = (float*) malloc(N * sizeof(float));    // Loss (CPU result)
    float *h_loss_gpu = (float*) malloc(N * sizeof(float));    // Loss (GPU result)

    // Initialize random logits and labels
    srand(time(0));
    for (int i = 0; i < N * C; i++) 
        h_logits[i] = (rand() % 100) / 10.0f - 5.0f;  // Random values between -5 and 5
    for (int i = 0; i < N; i++) 
        h_labels[i] = rand() % C;  // Random class index [0, C-1]

    // **CPU Execution**
    clock_t start_cpu = clock();
    softmaxCrossEntropyCPU(h_logits, h_labels, h_loss_cpu, N, C);
    clock_t end_cpu = clock();
    double time_cpu = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC * 1000.0;  // ms

    // **GPU Execution**
    float *d_logits, *d_loss;
    int *d_labels;
    cudaMalloc(&d_logits, N * C * sizeof(float));
    cudaMalloc(&d_labels, N * sizeof(int));
    cudaMalloc(&d_loss, N * sizeof(float));

    cudaMemcpy(d_logits, h_logits, N * C * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start_gpu, end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);

    cudaEventRecord(start_gpu);
    softmaxCrossEntropyGPU(d_logits, d_labels, d_loss);
    cudaEventRecord(end_gpu);
    cudaEventSynchronize(end_gpu);

    cudaMemcpy(h_loss_gpu, d_loss, N * sizeof(float), cudaMemcpyDeviceToHost);

    float time_gpu;
    cudaEventElapsedTime(&time_gpu, start_gpu, end_gpu);

    // **Comparison**
    float error = 0.0f;
    for (int i = 0; i < N; i++) 
        error += fabs(h_loss_cpu[i] - h_loss_gpu[i]);
    
    std::cout << "CPU Time: " << time_cpu << " ms\n";
    std::cout << "GPU Time: " << time_gpu << " ms\n";
    std::cout << "Loss Difference (Sum of Absolute Errors): " << error << "\n";

    // Free memory
    free(h_logits);
    free(h_labels);
    free(h_loss_cpu);
    free(h_loss_gpu);
    cudaFree(d_logits);
    cudaFree(d_labels);
    cudaFree(d_loss);

    return 0;
}

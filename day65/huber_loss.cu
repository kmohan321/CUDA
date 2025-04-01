#include<stdio.h>
#include<cuda.h>


__global__ void huber_loss(float *logits, float *labels, float *loss, int M, int N, float delta){

    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    
    if(row < M && col < N){

        float curr_logits = logits[row * N + col];
        float curr_labels = labels[row *N + col];

        float abs_distance = fabsf(curr_logits - curr_labels);
        if(abs_distance<delta){
            loss[row*N + col] = 0.5f * abs_distance * abs_distance;
        }
        else{
            loss[row * N + col] = delta * (abs_distance - 0.5f * delta);
        }
    }
}

void random_init(float *arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // Random values between -1 and 1
    }
}

int main() {

    int M = 32;
    int N = 32;
    float delta = 1.0f;
    int size = M * N * sizeof(float);

    float *h_logits = (float *)malloc(size);
    float *h_labels = (float *)malloc(size);
    float *h_loss = (float *)malloc(size);

    random_init(h_logits, M * N);
    random_init(h_labels, M * N);

    float *d_logits, *d_labels, *d_loss;
    cudaMalloc((void **)&d_logits, size);
    cudaMalloc((void **)&d_labels, size);
    cudaMalloc((void **)&d_loss, size);

    cudaMemcpy(d_logits, h_logits, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    huber_loss<<<gridSize, blockSize>>>(d_logits, d_labels, d_loss, M, N, delta);
    
    cudaMemcpy(h_loss, d_loss, size, cudaMemcpyDeviceToHost);

    printf("Logits\t\tLabels\t\tLoss\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            printf("%.4f\t%.4f\t%.4f\n", h_logits[idx], h_labels[idx], h_loss[idx]);
        }
    }

    free(h_logits);
    free(h_labels);
    free(h_loss);
    cudaFree(d_logits);
    cudaFree(d_labels);
    cudaFree(d_loss);

    return 0;
}





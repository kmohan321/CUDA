#include<stdio.h>
#include<cuda.h>

__global__ void sgd_momentum(float * gradient, float *weights, float *velocity, int M,int N, 
    float lr,float beta,int blocksize){


    int col  = blockDim.x * blockIdx.x + threadIdx.x % blocksize;
    int row  = blockDim.y * blockIdx.y + threadIdx.x / blocksize;

    if(row < M && col < N ){

        float velocity_t1 = beta * velocity[row *N + col] + (1 - beta) * gradient[row * N + col];
        weights[row *N + col] = weights[row *N + col] - lr * velocity_t1;
        velocity[row * N + col] = velocity_t1;
    }

}

int main() {
    
    int M = 4, N = 4; 
    int size = M * N * sizeof(float);

    float lr = 0.01f;
    float beta = 0.9f;

    float *h_weights = (float*)malloc(size);
    float *h_gradient = (float*)malloc(size);
    float *h_velocity = (float*)malloc(size);


    for (int i = 0; i < M * N; i++) {
        h_weights[i] = 1.0f;  
        h_gradient[i] = 0.1f;  
        h_velocity[i] = 0.0f;  
    }

    float *d_weights, *d_gradient, *d_velocity;
    cudaMalloc((void**)&d_weights, size);
    cudaMalloc((void**)&d_gradient, size);
    cudaMalloc((void**)&d_velocity, size);


    cudaMemcpy(d_weights, h_weights, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gradient, h_gradient, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocity, h_velocity, size, cudaMemcpyHostToDevice);

    int blocksize = 16; 
    dim3 blockSize(blocksize*blocksize);  
    dim3 grid((N + blocksize - 1) / blocksize, 
                 (M + blocksize - 1) / blocksize);

    sgd_momentum<<<grid, blockSize>>>(d_gradient, d_weights, d_velocity, M, N, lr, beta, blocksize);

    cudaMemcpy(h_weights, d_weights, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_velocity, d_velocity, size, cudaMemcpyDeviceToHost);

    printf("Updated Weights:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", h_weights[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_weights);
    cudaFree(d_gradient);
    cudaFree(d_velocity);

    free(h_weights);
    free(h_gradient);
    free(h_velocity);

    return 0;
}
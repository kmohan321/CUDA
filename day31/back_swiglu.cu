#include<stdio.h>
#include<cuda.h>
#include <math.h>  


__global__ void back_swiglu(float *grad_out, float *Wx, float *out, int r, int c){

    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    if(row<r && col <c){
      float curr_value = Wx[row*c + col];
      float sigmoid = 1.0f / (1.0f + expf(-curr_value));
      float grad_value = sigmoid *curr_value* (2.0f + curr_value *(1 - sigmoid));

      out[row*c + col] = grad_out[row*c + col] * grad_value;
    }
    
}

void initialize_matrix(float *matrix, int r, int c) {
    for (int i = 0; i < r * c; i++) {
        matrix[i] = ((float)rand() / RAND_MAX) * 2 - 1; // Random values between -1 and 1
    }
}

void print_matrix(float *matrix, int r, int c) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            printf("%0.4f ", matrix[i * c + j]);
        }
        printf("\n");
    }
}

int main() {
    int r = 4, c = 4;
    size_t size = r * c * sizeof(float);
    int num_threads = 16;

    float *h_grad_out = (float*)malloc(size);
    float *h_Wx = (float*)malloc(size);
    float *h_out = (float*)malloc(size);

    initialize_matrix(h_grad_out, r, c);
    initialize_matrix(h_Wx, r, c);

    float *d_grad_out, *d_Wx, *d_out;
    cudaMalloc((void**)&d_grad_out, size);
    cudaMalloc((void**)&d_Wx, size);
    cudaMalloc((void**)&d_out, size);

    cudaMemcpy(d_grad_out, h_grad_out, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Wx, h_Wx, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(num_threads, num_threads);
    dim3 numBlocks((c + num_threads - 1) / num_threads, (r + num_threads - 1) / num_threads);

    back_swiglu<<<numBlocks, threadsPerBlock>>>(d_grad_out, d_Wx, d_out, r, c);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    printf("Gradient Output:\n");
    print_matrix(h_grad_out, r, c);
    
    printf("\nWx Matrix:\n");
    print_matrix(h_Wx, r, c);
    
    printf("\nBackward SwigLU Output:\n");
    print_matrix(h_out, r, c);

    cudaFree(d_grad_out);
    cudaFree(d_Wx);
    cudaFree(d_out);
    free(h_grad_out);
    free(h_Wx);
    free(h_out);

    return 0;
}



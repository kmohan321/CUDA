#include <stdio.h>
#include <cuda.h>

#define TILE 32  // Tile size for shared memory optimization

__global__ void fused_projection_bce(float *input, float *weight, float *bias, 
                                     float *target, float *output, float *loss, 
                                     int batch_size, int input_dim, int output_dim) {
    
    extern __shared__ float smem[]; 

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.x * TILE + tx;
    int col = blockIdx.y * TILE + ty;

    float *input_tiled = smem;
    float *weight_tiled = smem + TILE * TILE;

    float sum = 0.0f;

    for (int i = 0; i < (input_dim + TILE - 1) / TILE; i++) {

        if (row < batch_size && (i * TILE + ty) < input_dim) {
            input_tiled[tx * TILE + ty] = input[row * input_dim + i * TILE + ty];
        } else {
            input_tiled[tx * TILE + ty] = 0.0f;
        }

        if (col < output_dim && (i * TILE + tx) < input_dim) {
            weight_tiled[tx * TILE + ty] = weight[(i * TILE + tx) * output_dim + col];
        } else {
            weight_tiled[tx * TILE + ty] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += input_tiled[tx * TILE + k] * weight_tiled[k * TILE + ty];
        }

        __syncthreads();
    }

    if (row < batch_size && col < output_dim) {
        float logits = sum + bias[col];
        float sigmoid = 1.0f / (1.0f + expf(-logits));
        output[row * output_dim + col] = sigmoid;

        // Compute BCE Loss
        float t = target[row * output_dim + col];
        float bce = -t * logf(sigmoid + 1e-9f) - (1 - t) * logf(1 - sigmoid + 1e-9f);
        loss[row * output_dim + col] = bce;
    }
}

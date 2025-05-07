#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 256
__device__ float squared_dist(const float* p, const float* q, int dims) {
    float acc = 0.0f;
    for (int j = 0; j < dims; ++j) {
        float delta = p[j] - q[j];
        acc += delta * delta;
    }
    return acc;
}

__global__ void loss_contrastive(
    const float* vec1, const float* vec2, const int* targets,
    float* result, int num_samples, int dims, float max_margin
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_samples) return;

    const float* p = vec1 + index * dims;
    const float* q = vec2 + index * dims;
    int target = targets[index];

    float distance_squared = squared_dist(p, q, dims);
    float final_loss;

    if (target == 1) {
        // Similar pair
        final_loss = distance_squared;
    } else {
        // Dissimilar pair
        float distance = sqrtf(distance_squared);
        float gap = fmaxf(max_margin - distance, 0.0f);
        final_loss = gap * gap;
    }

    result[index] = 0.5f * final_loss;
}

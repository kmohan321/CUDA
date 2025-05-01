#include <cuda_runtime.h>
#include <float.h>   
#include <stddef.h>  

__global__ void maxpool2d_kernel(const float* __restrict__ input,
                      int in_height, int in_width,
                      int kernel_size,
                      int stride,
                      int padding,
                      int dilation,
                      int out_height, int out_width,
                      float* __restrict__ output) {

    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_y >= out_height || out_x >= out_width) return;

    float max_value = -FLT_MAX;

    for (int kernel_y = 0; kernel_y < kernel_size; ++kernel_y) {
        int in_y = out_y * stride + kernel_y * dilation - padding;

        for (int kernel_x = 0; kernel_x < kernel_size; ++kernel_x) {
            int in_x = out_x * stride + kernel_x * dilation - padding;

            if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                float val = input[in_y * in_width + in_x];
                max_value = fmaxf(max_value, val);
            }
        }
    }

    output[out_y * out_width + out_x] = max_value;
}

extern "C" void solution(const float* input,
              int kernel_size,
              int stride,
              int padding,
              int dilation,
              float* output,
              size_t in_height,
              size_t in_width) {

    int out_height = ((int)in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width  = ((int)in_width  + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    dim3 block_dim(16, 16);
    dim3 grid_dim((out_width  + block_dim.x - 1) / block_dim.x,
                  (out_height + block_dim.y - 1) / block_dim.y);

    maxpool2d_kernel<<<grid_dim, block_dim>>>(
        input,
        (int)in_height,
        (int)in_width,
        kernel_size,
        stride,
        padding,
        dilation,
        out_height,
        out_width,
        output
    );
}

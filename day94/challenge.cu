#include <cuda_runtime.h>


__global__ void avgpool1d_forward(const float* __restrict__ src,
                       float* __restrict__ dst,
                       int input_length,
                       int pool_size,
                       int stride,
                       int pad,
                       int output_length) {
    
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx >= output_length) return;

    int pool_start = out_idx * stride - pad;
    float acc = 0.0f;

    for (int offset = 0; offset < pool_size; ++offset) {
        int in_idx = pool_start + offset;

        if (in_idx >= 0 && in_idx < input_length) {
            acc += src[in_idx];
        }
    }

    dst[out_idx] = acc / static_cast<float>(pool_size);
}

extern "C" void solution(const float* input_data,
              int pool_size,
              int stride,
              int pad,
              float* output_data,
              size_t input_len) {
    
    int output_len = static_cast<int>((input_len + 2 * pad - pool_size) / stride + 1);

    const int threads_per_block = 256;
    int total_blocks = (output_len + threads_per_block - 1) / threads_per_block;

    avgpool1d_forward<<<total_blocks, threads_per_block>>>(
        input_data,
        output_data,
        static_cast<int>(input_len),
        pool_size,
        stride,
        pad,
        output_len
    );
}

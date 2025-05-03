#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__constant__ float constKernel[16384];

__global__ void runConvolution(const float* __restrict__ input,
                               const float* __restrict__ kernel,
                               float* __restrict__ output,
                               int height, int width,
                               int kernelHeight, int kernelWidth,
                               int useConstMem)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height)
        return;

    float val = 0.0f;
    int padH = kernelHeight / 2;
    int padW = kernelWidth / 2;

    int totalKernelSize = kernelHeight * kernelWidth;

    bool smallKernel = (kernelHeight <= 7);

    if (smallKernel) {
        #pragma unroll
        for (int dy = -padH; dy <= padH; ++dy) {
            for (int dx = -padW; dx <= padW; ++dx) {
                int iy = y + dy;
                int ix = x + dx;
                if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                    int kIdx = (dy + padH) * kernelWidth + (dx + padW);
                    float kVal = useConstMem ? constKernel[kIdx] : kernel[kIdx];
                    val += input[iy * width + ix] * kVal;
                }
            }
        }
    } else {
        for (int dy = -padH; dy <= padH; ++dy) {
            for (int dx = -padW; dx <= padW; ++dx) {
                int iy = y + dy;
                int ix = x + dx;
                if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                    int kIdx = (dy + padH) * kernelWidth + (dx + padW);
                    float kVal = useConstMem ? constKernel[kIdx] : kernel[kIdx];
                    val += input[iy * width + ix] * kVal;
                }
            }
        }
    }

    output[y * width + x] = val;
}

extern "C" void solution(const float* input,
              const float* kernel,
              float* output,
              size_t h, size_t w,
              size_t kh, size_t kw)
{
    bool canUseConst = (kh * kw <= 16384);
    if (canUseConst) {
        cudaMemcpyToSymbol(constKernel, kernel, kh * kw * sizeof(float));
    }

    dim3 threads(32, 8);
    dim3 blocks((w + threads.x - 1) / threads.x,
                (h + threads.y - 1) / threads.y);

    runConvolution<<<blocks, threads>>>(
        input, kernel, output,
        static_cast<int>(h), static_cast<int>(w),
        static_cast<int>(kh), static_cast<int>(kw),
        canUseConst ? 1 : 0
    );

    cudaDeviceSynchronize();
}

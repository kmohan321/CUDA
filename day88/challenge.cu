#include <cuda_runtime.h>
#define PIE 3.14159

__global__ void GELU(const float *__restrict__ input, float *__restrict__ output, int N, int M){
    
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    // int row = threadIdx.y + blockDim.y  * blockIdx.y;

    int offset = idx;

    float constant_value = sqrtf(2.0f / PIE);

    float4 value = *(reinterpret_cast<const float4*>(input) + offset);

    float curr_value_x = value.x;
    float curr_value_y = value.y;
    float curr_value_z = value.z;
    float curr_value_w = value.w;
    float value_3_x = curr_value_x * curr_value_x * curr_value_x;
    float value_3_y = curr_value_y * curr_value_y * curr_value_y;
    float value_3_z = curr_value_z * curr_value_z * curr_value_z;
    float value_3_w = curr_value_w * curr_value_w * curr_value_w;

    float tanh_value_x = __tanhf(constant_value * fmaf(0.044715f, value_3_x, curr_value_x));
    float tanh_value_y = __tanhf(constant_value * fmaf(0.044715f, value_3_y, curr_value_y));
    float tanh_value_z = __tanhf(constant_value * fmaf(0.044715f, value_3_z, curr_value_z));
    float tanh_value_w = __tanhf(constant_value * fmaf(0.044715f, value_3_w, curr_value_w));

    float4 out;
    out.x = 0.5 * curr_value_x * (1.0f + tanh_value_x);
    out.y = 0.5 * curr_value_y * (1.0f + tanh_value_y);
    out.z = 0.5 * curr_value_z * (1.0f + tanh_value_z);
    out.w = 0.5 * curr_value_w * (1.0f + tanh_value_w);

    *(reinterpret_cast<float4*>(output)+offset) = out;

}
// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {  

    dim3 blocksize(256,1);
    // dim3 grid( (n /4 + blocksize.x -1)/blocksize.x, (m + blocksize.y -1)/blocksize.y) ;
    dim3 grid( (n /4 * m + blocksize.x -1)/blocksize.x) ;
    GELU<<<grid, blocksize>>>(input, output, n,m);  
}
#include <cuda_runtime.h>

//let's try loading multiple elements
__global__ void sigmoid(const float * __restrict__ input, float * __restrict__ output, int M, int N){
    
        int x = threadIdx.x + blockDim.x * blockIdx.x; //col
        int y = threadIdx.y + blockDim.y * blockIdx.y; //row 

        if(y < M && x < N/4){
            const int idx = y * N/4 + x;
            float4 curr_val = __ldg((reinterpret_cast<const float4*>(input) + idx));
            
            float4 out;
            out.x = __frcp_rn(1.0f + __expf(-curr_val.x));
            out.y = __frcp_rn(1.0f + __expf(-curr_val.y));
            out.z = __frcp_rn(1.0f + __expf(-curr_val.z));
            out.w = __frcp_rn(1.0f + __expf(-curr_val.w));
            
            reinterpret_cast<float4*>(output)[idx] = out;
         }
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {    
    int N=n, M=m;
    dim3 blocksize(32,32);
    dim3 grid ((N/4 + blocksize.x - 1)/blocksize.x,(M + blocksize.y - 1)/blocksize.y);
    sigmoid<<<grid, blocksize>>>(input,output,m,n);

}

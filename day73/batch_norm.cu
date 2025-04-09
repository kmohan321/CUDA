#include <cuda_runtime.h>

//simply run parallely over the feature dimension 
__global__ void batch_normalised(const float *input, float *output,int B, int F, int D1, int D2){

    
    int feature_idx = blockIdx.x;
    int idx = threadIdx.x; //for h*w 

    if(feature_idx>=F) return;

    extern __shared__ float smem[];
    float *sum_smem = smem;
    float *sq_smem = sum_smem + blockDim.x;

    float sum = 0.0f;
    float sq_sum = 0.0f;
    for(int i =0; i <B; i++){
        for(int j =idx; j<D1*D2; j += blockDim.x){
            float curr_value = input[i * F*D1*D2 + feature_idx * D1*D2 + j];
            sum += curr_value;
            sq_sum += curr_value * curr_value;
        }
    }

    sum_smem[idx] = sum;
    sq_smem[idx] = sq_sum;
    __syncthreads();
    for(int i = blockDim.x/2; i>0; i/=2){
        if(idx<i){
            sum_smem[idx] += sum_smem[idx + i];
            sq_smem[idx] += sq_smem[idx + i];

        }
        __syncthreads();
        
    }
    int factor = (B*D1*D2);
    float mean = sum_smem[0]/factor;
    float variance = sq_smem[0]/factor - (mean * mean);
    float r_variance = 1.0f / sqrtf(variance + 1e-5);

    for(int i =0; i <B; i++){
        for(int j = idx; j<D1*D2; j += blockDim.x){
            float curr_value = input[i * F*D1*D2 + feature_idx * D1*D2 + j];
            output[i * F*D1*D2 + feature_idx * D1*D2 + j] = (curr_value - mean) * r_variance;
        }
    }

}

// Note: X, Y are all device pointers to float32 arrays
extern "C" void solution(const float* X, float* Y, size_t B, size_t F, size_t D1, size_t D2) {    

    int threads = 1024;
    dim3 blocksize(threads);
    dim3 grid(F);
    int smem_size = 2*threads*sizeof(float);
    batch_normalised<<<grid,blocksize,smem_size>>>(X,Y,B,F,D1,D2);


}
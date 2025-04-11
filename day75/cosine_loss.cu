#include <cuda_runtime.h>

__global__ void cosine_sim(const float *prediction, const float *targets,float *loss, int N, int D){


    int batch_idx = blockIdx.x;
    int idx = threadIdx.x;
    extern __shared__ float smem[];
    float *A = smem;
    float *B = smem + blockDim.x;
    float *C = B + 2*blockDim.x;

    float local_sum = 0.0f;
    float d1 = 0.0f;
    float d2 = 0.0f;
    for(int i = idx ; i < D; i+= blockDim.x){
        float curr_pred = prediction[batch_idx * D + i];
        float curr_target = targets[batch_idx * D + i];
        local_sum += curr_pred * curr_target;
        d1 += curr_pred * curr_pred;
        d2 += curr_target * curr_target;
    }

    A[idx] = local_sum;
    B[idx] = d1;
    C[idx] = d2;

    __syncthreads();
    for(int i = blockDim.x/2; i>0; i/=2){
        if(idx<i){
            A[idx] += A[idx+i];
            B[idx] += B[idx+i];
            C[idx] += C[idx+i];
        }
        __syncthreads();
    }
    float sum = A[0];
    float norm1 = max(1e-8,sqrtf(B[0]));
    float norm2 = max(1e-8,sqrtf(C[0]));

    if(idx ==0){
        loss[batch_idx] = 1 - (sum / (norm1 * norm2));
    }


}



// Note: predictions, targets, output are all device pointers to float32 arrays
extern "C" void solution(const float* predictions, const float* targets, float* output, size_t n, size_t d) {    

    int threads = 512;
    dim3 blocksize(threads);
    dim3 grid(n);
    int smem_size = 3* threads * sizeof(float);
    cosine_sim<<<grid,blocksize,smem_size>>>(predictions,targets,output,n,d);

}
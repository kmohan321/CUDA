#include <cuda_runtime.h>

__global__ void Triplet_Loss (const float*anchor, const float*positive, const float*negative, float *loss, int B, int E, float margin){

       int idx = threadIdx.x;
       int batch = blockIdx.x;

        if(batch>=B) return;

        extern __shared__ float smem[];
        float *smem1 = smem;
        float *smem2 = smem + blockDim.x;

        float local_d1 = 0.0f;
        float local_d2 = 0.0f;
        for(int i = idx; i < E; i += blockDim.x){
            float curr_anchor = anchor[batch * E + i];
            float curr_positive = positive[batch * E + i];
            float curr_negative = negative[batch * E + i];
            local_d1 += (curr_anchor - curr_positive) * (curr_anchor - curr_positive);
            local_d2 += (curr_anchor - curr_negative) * (curr_anchor - curr_negative) ; 
        }
        
        smem1[idx] = local_d1;
        smem2[idx] = local_d2;
        __syncthreads();
        //reduction time
        for(int i = blockDim.x/2; i>0; i /=2){
            if(idx<i){
                smem1[idx] += smem2[idx+i];
                smem2[idx] += smem2[idx+i];
            }
            __syncthreads();
        }

        float d1 = sqrtf(smem1[0]);
        float d2 = sqrtf(smem2[0]);

        if(idx==0){

        }

        


}


// Note: anchor, positive, negative, loss are all device pointers to float32 arrays
extern "C" void solution(const float* anchor, const float* positive, const float* negative, float* loss, size_t B, size_t E, float margin) {    
}
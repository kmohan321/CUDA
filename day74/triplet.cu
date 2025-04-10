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
                smem1[idx] += smem1[idx+i];
                smem2[idx] += smem2[idx+i];
            }
            __syncthreads();
        }

        float d1 = sqrtf(smem1[0]);
        float d2 = sqrtf(smem2[0]);

        if(idx==0){
            float value = d1 - d2 + margin;
            loss[batch] = fmaxf(0,value);
        }

}

__global__ void block_sum(float *block_sum, float *loss, int B){
    int idx = threadIdx.x;

    extern __shared__ float smem[];
    float local_sum = 0.0f;
    for(int i = idx; i<B; i+=blockDim.x){
        local_sum += block_sum[i];
    }

    smem[idx] = local_sum;
    for(int i = blockDim.x/2; i>0; i /=2){
    if(idx<i){
        smem[idx] += smem[idx+i];
    }
    __syncthreads();
    }
    
    if(idx==0){
        float sum = smem[0];
        *loss = sum/B;
    }
    
}


// Note: anchor, positive, negative, loss are all device pointers to float32 arrays
extern "C" void solution(const float* anchor, const float* positive, const float* negative, float* loss, size_t B, size_t E, float margin) {    

    int threads = 1024;
    dim3 blocksize(threads);
    dim3 grid(B);
    int smem_size = 2*threads*sizeof(float);
    int smem_size2 = threads * sizeof(float);

    float *d_avg;
    cudaMalloc((void**)&d_avg , B*sizeof(float));
    Triplet_Loss<<<grid,blocksize,smem_size>>>(anchor,positive,negative,d_avg, B,E,margin);
    
    block_sum<<<1,blocksize,smem_size2>>>(d_avg,loss,B);
}
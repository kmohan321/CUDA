#include <cuda_runtime.h>


__global__ void MSE(const float * __restrict__ predictions, const float *__restrict__ targets , float *__restrict__ d_block, size_t N){

    int row = blockIdx.x;
    int idx = threadIdx.x;

    extern __shared__ float smem[];

    float local_sum = 0.0f;
    for(int i = idx; i<N; i += blockDim.x){
        float curr_pred = predictions[row * N + i];
        float curr_target = targets[row * N + i];
        float diff = curr_pred - curr_target;
        local_sum += diff * diff;
    }
    smem[idx] = local_sum;
    __syncthreads();

    //reduction 
    for(int stride = blockDim.x/2; stride>0; stride /=2){
        if(idx <stride ){
            smem[idx] += smem[idx + stride];
        }
        __syncthreads();
    }
    if(idx==0){
        d_block[row] = smem[0];
    }

}

__global__ void BlockSum(float *input, float *output, size_t N, int blocks){

    extern __shared__ float smem[];
    int idx = threadIdx.x;
    smem[idx] = 0.0f;
    float block_local_sum = 0.0f;
    for(int i = idx; i< blocks; i += blockDim.x){
        block_local_sum += input[i];
    }

    smem[idx] = block_local_sum;
    __syncthreads();

    for(int stride = blockDim.x/2; stride>0; stride /=2){
        if(idx < stride){
            smem[idx] += smem[idx + stride];
        }
        __syncthreads();
    }
    if(idx==0){
        output[0] = smem[0] / (N * blocks);
    }
    
}


// Note: predictions, targets, output, shape are all device pointers to float32 arrays
extern "C" void solution(const float* predictions, const float* targets, float* output, size_t* shape, size_t ndim) {    

    size_t *h_shape = (size_t*)malloc(ndim * sizeof(size_t));
    cudaMemcpy(h_shape, shape, ndim * sizeof(size_t), cudaMemcpyDeviceToHost);
    int threads = 64;
    dim3 blocksize(threads);
    size_t batch_size = h_shape[0];
    dim3 grid(batch_size);
    int smem_size = threads * sizeof(float);
    float *d_block;
    cudaMalloc((void**)&d_block, h_shape[0] * sizeof(float));
    size_t N = 1;
    for(int i = 1 ; i < ndim; i++){
        N *= h_shape[i];
    }
    MSE<<<grid, blocksize, smem_size>>>(predictions, targets, d_block, N);
    cudaDeviceSynchronize();

    BlockSum<<<1, blocksize, smem_size>>>(d_block, output, N, batch_size);
    cudaDeviceSynchronize();
    
    cudaFree(d_block);
    free(h_shape);
}
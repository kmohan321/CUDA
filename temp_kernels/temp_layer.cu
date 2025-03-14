#include<stdio.h>
#include<cuda.h>

#define M_PI 3.14159265f

#define blocksize 1024
#define r 1024
#define c 65536

/*we will use each block to process entire row */
__global__ void soft_opt(float* __restrict__ xd, float* __restrict__ resd) {
    // max and norm reduction will happen in shared memory (static)
    __shared__ double smem[1024];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    // edge condition (we don't process further)
    if (row >= r) return;

    float* input_row = xd + row * c;
    float* output_row = resd + row * c;
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    // compute local max and norm for each thread
    // and then finally have a sync barrier before moving on
    for (int i = tid; i < c; i += blockDim.x) {
        float x = input_row[i];
        if (x > local_max) {
            local_norm *= expf(local_max - x);
            local_max = x;
        }
        local_norm += expf(x - local_max);
    }
    __syncthreads();

    // each thread will have its own local max
    // we store it in the tid of the shared memory
    smem[tid] = local_max;
    __syncthreads();

    // block-level reduction in O(log(c)) time over all threads
    // is faster than linear reduction over all threads
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            smem[tid] = fmax(smem[tid], smem[tid + stride]);
        }
        // sync barrier before next iteration to ensure correctness
        __syncthreads();
    }

    // the first element after max reduction from all threads
    // will contain the global max for the row
    float row_max = smem[0];
    __syncthreads();

    // each thread will have its own local norm
    // we will store the corrected local norm in the shared memory
    // again, exploits property of exponentials
    smem[tid] = local_norm * expf(local_max - row_max);
    __syncthreads();

    // sum reduction similar to above for global norm factor
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    float row_norm = smem[0];
    __syncthreads();

    // finally, compute softmax
    for (int i = tid; i < c; i += blockDim.x) {
        output_row[i] = expf(input_row[i] - row_max) / row_norm;
    }
}

float random_normal_clamped(float min, float max) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float num = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    if (num < min)
        return min;
    if (num > max)
        return max;
    return num;
}
int main(){

    float *input = (float*)malloc(r * c * sizeof(float));
    float *output = (float*)malloc(r * c * sizeof(float));

    float *in, *out;

    cudaMalloc((void**)&in ,r*c*sizeof(float));
    cudaMalloc((void**)&out , r*c*sizeof(float));

    //let's intialize the matrix
    for(int i =0; i<r;i++){
      for(int j =0; j<c;j++){
        input[j + i*c] = random_normal_clamped(-10,10);
      }
    }

    cudaMemcpy(in,input,r*c*sizeof(float),cudaMemcpyHostToDevice);

    dim3 gridsize(r);
    dim3 Block_Size(blocksize);

    soft_opt<<<gridsize,Block_Size>>>(in,out);

    cudaMemcpy(output,out,r*c*sizeof(float),cudaMemcpyDeviceToHost);

    // for debugging 
    for(int i =0; i<r;i++){
      float sum = 0.0f;
      for(int j =0; j<c;j++){
        sum += output[j+i*c];
      }
      printf("\nsum is %f \n",sum);
    }

    cudaFree(in);
    cudaFree(out);
    free(input);
    free(output);

}
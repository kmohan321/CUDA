#include<stdio.h>
#include<cuda.h>

#define m 24
#define threads_per_block 24


//important kernel 
__global__ void vec_dot(float *v1, float *v2, float *v3){
      
      int global_idx = threadIdx.x + blockDim.x * blockIdx.x;
      int s_idx = threadIdx.x; //id for thread in a block

      __shared__ float s[threads_per_block]; //storing the partial dot product in shared_memory

      if(global_idx < m){
          s[s_idx] = v1[global_idx] * v2[global_idx];
      }
      __syncthreads(); //ensuring each thread has written value in shared_memory

      //partial reduction sum for shared_memory 
    for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if(s_idx < stride && (global_idx + stride) < m){
            s[s_idx] += s[s_idx + stride];
        }
    }
      __syncthreads(); //ensuring that the each threads completes its work

      if (s_idx == 0){ //storing the values from shared memory to the global_memory
        v3[blockIdx.x] = s[0];
      }      
    }


int main(){

    int blocks = (m + threads_per_block -1) / threads_per_block;
    dim3 grid_size (blocks);
    dim3 block_size (threads_per_block);

    float *vec1 = (float*)malloc(m * sizeof(float));
    float *vec2 = (float*)malloc(m * sizeof(float));
    float *vec3 = (float*)malloc( blocks * sizeof(float));

    float *p1, *p2, *p3;
    cudaMalloc((void**)&p1, m * sizeof(float));
    cudaMalloc((void**)&p2, m * sizeof(float));
    cudaMalloc((void**)&p3, blocks * sizeof(float));

    //intializing the vectors
    for(int i = 0;i<m;i++){
      vec1[i] = i ;
      vec2[i] = i * 2 ;
    }

    cudaMemcpy(p1,vec1,m * sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(p2,vec2,m * sizeof(float),cudaMemcpyHostToDevice);

    vec_dot<<<grid_size,block_size>>>(p1,p2,p3);

    cudaMemcpy(vec3,p3,blocks * sizeof(float),cudaMemcpyDeviceToHost);
    //completing the summation
    float sum = 0 ;
    for(int i =0; i<blocks; i++){
      sum += vec3[i];
    }
    printf("the dot product is %.2f",sum);

    // Free memory
    cudaFree(p1);
    cudaFree(p2);
    cudaFree(p3);
    free(vec1);
    free(vec2);
    free(vec3);

  return 0;
}
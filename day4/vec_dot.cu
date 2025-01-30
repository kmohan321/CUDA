#include<stdio.h>
#include<cuda.h>

#define m 1291
#define threads_per_block 128


//important kernel 
__global__ void vec_dot(int *v1, int *v2, int *v3){
      
      int global_idx = threadIdx.x + blockDim.x * blockIdx.x;
      int s_idx = threadIdx.x; //id for thread in a block

      __shared__ int s[threads_per_block]; //storing the partial dot product in shared_memory
      //(key point whole shared memory is now intialized with int 0)
      if(global_idx < m){
          s[s_idx] = v1[global_idx] * v2[global_idx];
      }
      __syncthreads(); //ensuring each thread has written value in shared_memory

      //partial reduction sum for shared_memory 
      //(works well when each block's shared memory has power of 2 filled positions)
    for(int stride = blockDim.x/2; stride > 0; stride >>= 1){
        if(s_idx < stride){
            s[s_idx] += s[s_idx + stride];
        }
        __syncthreads(); //ensuring that the each threads completes its work
    }
      if (s_idx == 0){ //storing the values from shared memory to the global_memory
        v3[blockIdx.x] = s[0];
      }      
    }


int main(){

    int blocks = (m + threads_per_block -1) / threads_per_block;
    dim3 grid_size (blocks);
    dim3 block_size (threads_per_block);

    int *vec1 = (int*)malloc(m * sizeof(int));
    int *vec2 = (int*)malloc(m * sizeof(int));
    int *vec3 = (int*)malloc( blocks * sizeof(int));

    int *p1, *p2, *p3;
    cudaMalloc((void**)&p1, m * sizeof(int));
    cudaMalloc((void**)&p2, m * sizeof(int));
    cudaMalloc((void**)&p3, blocks * sizeof(int));

    //intializing the vectors
    for(int i = 0;i<m;i++){
      vec1[i] = i ;
      vec2[i] = i * 2 ;
    }

    cudaMemcpy(p1,vec1,m * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(p2,vec2,m * sizeof(int),cudaMemcpyHostToDevice);

    vec_dot<<<grid_size,block_size>>>(p1,p2,p3);

    cudaMemcpy(vec3,p3,blocks * sizeof(int),cudaMemcpyDeviceToHost);
    //completing the summation
    int sum = 0 ;
    for(int i =0; i<blocks; i++){
      sum += vec3[i];
    }
    printf("the dot product is %d",sum);

    // Free memory
    cudaFree(p1);
    cudaFree(p2);
    cudaFree(p3);
    free(vec1);
    free(vec2);
    free(vec3);

  return 0;
}
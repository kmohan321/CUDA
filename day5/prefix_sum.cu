#include <stdio.h>
#include <cuda.h>

#define m 512
#define threads_per_block 512

__global__ void prefix_sum(int *v1, int *v2) {
    int g_idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idx = threadIdx.x;

    __shared__ int s[threads_per_block];

    // Load data into shared memory
    if (g_idx < m) {
        s[idx] = v1[g_idx];
    } else {
        s[idx] = 0; 
    }
    
    __syncthreads();

    for(int stride =1;stride<blockDim.x;stride*=2){

      int sum;
      if(idx >=stride){
        sum = s[idx] + s[idx-stride];

      }
      __syncthreads();
      if(idx>=stride){
        s[idx] = sum;
      }
    }
    if (idx<m){
      v2[g_idx] = s[idx];
    }

}


int main() {
    int blocks = (m + threads_per_block - 1) / threads_per_block;
    dim3 grid_size(blocks);
    dim3 block_size(threads_per_block);

    int *vec1 = (int*)malloc(m * sizeof(int));
    int *vec2 = (int*)malloc(m * sizeof(int));

    int *p1, *p2;
    cudaMalloc((void**)&p1, m * sizeof(int));
    cudaMalloc((void**)&p2, m * sizeof(int));

    // Initializing the vector
    for (int i = 0; i < m; i++) {
        vec1[i] = i + 1;  
    }

    cudaMemcpy(p1, vec1, m * sizeof(int), cudaMemcpyHostToDevice);

    prefix_sum<<<grid_size, block_size>>>(p1, p2);

    cudaMemcpy(vec2, p2, m * sizeof(int), cudaMemcpyDeviceToHost); 

    // Displaying the prefix sum
    printf("Prefix Sum Output:\n");
    for (int i = 0; i < m; i++) {
        printf("%d ", vec2[i]);
    }
    printf("\n");

    // Free memory
    cudaFree(p1);
    cudaFree(p2);
    free(vec1);
    free(vec2);

    return 0;
}

#include<stdio.h>
#include<cuda.h>

#define M 100000
#define m 256 //threads per block

// this time global indexing
__global__ void vec_add(int *v1, int *v2, int *v3){

    __shared__ int s1[m];
    __shared__ int s2[m];

    int idx = threadIdx.x;
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;

    if(gidx < M){
        s1[idx] = v1[gidx]; //filling the shared memory 
        s2[idx] = v2[gidx];  
    }

    __syncthreads();

    if(gidx < M){
        v3[gidx] = s2[idx] + s1[idx];
    }   
}

int main(){

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int vec1[M], vec2[M], vec3[M]; //arrays allocated on the host 
    int *p1, int *p2, int *p3;  //pointers for memory assigned on the gpu

    float memAlloc_time;
    cudaEventRecord(start);
    // pointers pointing to memory on gpu
    cudaMalloc((void**)&p1, M * sizeof(int));
    cudaMalloc((void**)&p2, M * sizeof(int));
    cudaMalloc((void**)&p3, M * sizeof(int));
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&memAlloc_time, start, stop);
    printf("Memory allocation time: %.3f ms\n", memAlloc_time);
    // let's assign some value to vectors
    for (int i=0; i<M; i++){
        vec1[i] = i;
        vec2[i] = i * 2;
    }
    float H2D_time;
    cudaEventRecord(start);
    //copying these vectors to gpu memory
    cudaMemcpy(p1,vec1,M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(p2,vec2,M * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&H2D_time, start, stop);
    printf("Host to Device transfer time: %.3f ms\n", H2D_time);

    // let's use multilple blocks
    // int blocks = (M + m -1)/m; //all elements should be covered
    int blocks = 8*20; //all elements should be covered
    int minGridSize, blockSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, vec_add, 0, 0);
    // printf("%d,%d\n",minGridSize,blockSize);
    // let's call the kernel 

    float kernel_time;
    cudaEventRecord(start);
    vec_add<<<blocks,m>>>(p1,p2,p3);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&kernel_time, start, stop);
    printf("Kernel execution time: %.3f ms\n", kernel_time);

    float D2H_time;
    cudaEventRecord(start);
    cudaMemcpy(vec3,p3,M * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&D2H_time, start, stop);
    printf("Device to Host transfer time: %.3f ms\n", D2H_time);
    
    // Calculate throughput
    float total_time = memAlloc_time + H2D_time + kernel_time + D2H_time;
    float bandwidth = (3 * M * sizeof(int)) / (total_time * 1e6); // GB/s
    printf("\nPerformance metrics:\n");
    printf("Total time: %.3f ms\n", total_time);
    printf("Effective Bandwidth: %.2f GB/s\n", bandwidth);

    printf("printing the output vector\n");

    for(int i= 0; i<5; i++){
        printf("%d + %d = %d\n", vec1[i], vec2[i], vec3[i]);
    }
    
    cudaFree(p1);
    cudaFree(p2);
    cudaFree(p3);
    
    return 0;
}



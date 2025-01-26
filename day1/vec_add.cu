#include<stdio.h>
#include<cuda.h>

#define M 10000

__global__ void vec_add(int *v1, int *v2, int *v3){

      v3[threadIdx.x] = v2[threadIdx.x] + v1[threadIdx.x];
}

int main(){

    int vec1[M], vec2[M], vec3[M]; //arrays allocated on the host 
    int *p1, int *p2, int *p3;  //pointers for memory assigned on the gpu

    // pointers pointing to memory on gpu
    cudaMalloc((void**)&p1, M * sizeof(int));
    cudaMalloc((void**)&p2, M * sizeof(int));
    cudaMalloc((void**)&p3, M * sizeof(int));

    // let's assign some value to vectors
    for (int i=0; i<M; i++){
        vec1[i] = i;
        vec2[i] = i * 2;
    }

    //copying these vectors to gpu memory
    cudaMemcpy(p1,vec1,M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(p2,vec2,M * sizeof(int), cudaMemcpyHostToDevice);

    // let's call the kernel 
    vec_add<<<1,M>>>(p1,p2,p3);

    cudaMemcpy(vec3,p3,M * sizeof(int), cudaMemcpyDeviceToHost);

    printf("printing the output vector\n");

    for(int i= 0; i<M; i++){
        printf("%d + %d = %d\n", vec1[i], vec2[i], vec3[i]);
    }
}




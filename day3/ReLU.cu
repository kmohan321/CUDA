#include<stdio.h>
#include<cuda.h>

#define r 1000
#define threads 256

__global__ void matrix_mul(float *arr1, float *arr2){

      int idx = threadIdx.x + blockDim.x * blockIdx.x;
      if(idx < r){
          arr2[idx] = fmaxf(0,arr1[idx]);
      }
}

int main(){
      
      float *arr1 = (float*)malloc(r*sizeof(float));
      float *arr2 = (float*)malloc(r*sizeof(float));

      float *p1, *p2;
      cudaMalloc((void**)&p1, r*sizeof(float));
      cudaMalloc((void**)&p2, r*sizeof(float));

      for(int i =0; i<r; i++){  
          if(i%2==0){
            arr1[i] = -i*2.1;
          }
          else{
            arr1[i] = i*.3;
          }
      }
      printf("vector... \n");
      for (int i = 0; i < 5; i++) {
          printf("%.2f\n",arr1[i]);
      }

      cudaMemcpy(p1,arr1,r*sizeof(float),cudaMemcpyHostToDevice);

      dim3 blocks(ceil(r/threads));
      dim3 block_size(threads);
      
      //kernel time
      matrix_mul<<<blocks , block_size>>>(p1,p2);
      cudaMemcpy(arr2,p2,r*sizeof(float),cudaMemcpyDeviceToHost);

      //visualizing the output
      printf("final vector. \n");
      for (int i = 0; i < 5; i++) {
          printf("%.2f\n",arr2[i]);
      }

      cudaFree(p1);
      cudaFree(p2);
      free(arr1);
      free(arr2);

      return 0;
}
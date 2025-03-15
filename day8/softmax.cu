#include<stdio.h>
#include<cuda.h>

#define BlockSize 32
#define Batch 64
#define VecDim 128

__global__ void softmax(float *input, float *output, float *max_array, float *sum_array,int blocky){

      __shared__ float s1[BlockSize*BlockSize];
      __shared__ float s2[BlockSize*BlockSize];
      __shared__ float s3[BlockSize];
      __shared__ float s4[BlockSize];

      int idx = threadIdx.x + blockDim.x * blockIdx.x; //batch dimension 
      int idy = threadIdx.y + blockDim.y * blockIdx.y; //vector dimension

      int x = threadIdx.x;
      int y = threadIdx.y;

      if(idx < Batch && idy < VecDim){
        s1[y + x*BlockSize] = input[idy + idx*VecDim]; //max value
        s2[y + x*BlockSize] = input[idy + idx*VecDim]; //sum value
      }
      __syncthreads();
      

      for(int stride = 1 ; stride <= blockDim.y/2; stride *=2) {
          if(y % (stride*2)== 0 && idx < Batch) {
              s1[y + x*BlockSize] = fmax(s1[y + x*BlockSize], s1[y + x*BlockSize + stride]);
          }
          __syncthreads();
      } 

      if (threadIdx.y == 0 && idx < Batch) {
          max_array[blockIdx.y + idx * blocky] = s1[y + x*BlockSize];
      }
      __syncthreads();

      if(idx < Batch) { //this can be done by only one thread in y if required 
        float maxVal = max_array[0 + idx * blocky];  
        for(int j = 1; j < blocky; j++) {           
            maxVal = fmax(maxVal, max_array[j + idx * blocky]);
        }
        s3[x] = maxVal;
      }
      __syncthreads();

      //exponentials of all values
      if(idx < Batch && idy < VecDim) {
          s2[y + x*BlockSize] = expf(s2[y + x*BlockSize]-s3[x]); //removed max value
      }
      __syncthreads();

      //parallel reduction for norm calculation
      for(int stride = 1; stride <= blockDim.y/2; stride *= 2) {
          if(y % (stride*2)== 0 && idx < Batch) {
              s2[y + x*BlockSize] += s2[y + x*BlockSize + stride];
          }
          __syncthreads();
      }

      //putting value in sum_array shape  ->(Batch,blocks in y)
      if (threadIdx.y == 0 && idx < Batch) {
          sum_array[blockIdx.y + idx * blocky] = s2[y + x*BlockSize];
      }
      __syncthreads();

      //for each element in batch, calculate the sum
      if(idx < Batch && threadIdx.y==0) {
          float sum = 0;
          for(int j = 0; j < blocky; j++) {     
              sum += sum_array[j + idx * blocky];
          }
          s4[x] = sum;
      }
      __syncthreads();

      //calculate the softmax
      if(idx<Batch && idy <VecDim){
          output[idy + idx * VecDim] = expf(input[idy + idx*VecDim]-s3[x])/s4[x]; 
      }
}

// blocks in x direction (Batch /blocksize) 
// blocks in y direction (vecdim/blocksize)

// blocksize in x direction (blocksize)
// blocksize in y direction (blocksize)

int main(){
    int blockx = ceil((float)Batch/BlockSize);
    int blocky = ceil((float)VecDim/BlockSize);

    float *input = (float*)malloc(Batch * VecDim * sizeof(float));
    float *output = (float*)malloc(Batch * VecDim * sizeof(float));

    float *in, *out, *max_array, *sum_array;

    cudaMalloc((void**)&in , Batch*VecDim*sizeof(float));
    cudaMalloc((void**)&out , Batch*VecDim*sizeof(float));
    cudaMalloc((void**)&max_array , Batch*blocky*sizeof(float));
    cudaMalloc((void**)&sum_array , Batch*blocky*sizeof(float));

    //let's intialize the matrix
    for(int i =0; i<Batch;i++){
      for(int j =0; j<VecDim;j++){
        input[j + i*VecDim] = (float)rand() / RAND_MAX;
        printf("%.2f ",input[j + i*VecDim]);
      }
      printf("\n");
    }

    cudaMemcpy(in,input,Batch*VecDim*sizeof(float),cudaMemcpyHostToDevice);

    dim3 gridsize(blockx,blocky);
    dim3 blocksize(BlockSize,BlockSize);

    softmax<<<gridsize,blocksize>>>(in,out,max_array,sum_array,blocky);

    cudaMemcpy(output,out,Batch*VecDim*sizeof(float),cudaMemcpyDeviceToHost);


    for(int i =0; i<Batch;i++){
      float sum = 0.0f;
      for(int j =0; j<VecDim;j++){
        // printf("%.2f ",output[j+i*VecDim]);
        sum += output[j+i*VecDim];
      }
      printf("\nsum is %.2f \n",sum);
    }

    cudaFree(in);
    cudaFree(out);
    free(input);
    free(output);

}
#include<stdio.h>
#include<cuda.h>

#define tile 4
#define r 10
#define c 10
#define blocksize 4

__global__ void matrix_tranpose(float *input, float *output){
      
      int idx = threadIdx.x + tile * blockIdx.x;
      int idy = threadIdx.y + tile * blockIdx.y;
      int x = threadIdx.x;
      int y = threadIdx.y;

      //intializing the shared memory
      __shared__ float s[tile*tile];

      //load data into shared memory 
      if(idx < r && idy <c){
        s[y + x*tile] = input[idy + idx*c];
      }
      __syncthreads();

      // transpose the matrix
      if(idx < c && idy < r){
        output[idx + idy*r] = s[y + x*tile];
      }
}

int main(){

    float *input = (float*)malloc(r * c * sizeof(float));
    float *output = (float*)malloc(r * c * sizeof(float));

    float *in, *out;

    cudaMalloc((void**)&in , r*c*sizeof(float));
    cudaMalloc((void**)&out , r*c*sizeof(float));

    //let's intialize the matrix
    for(int i =0; i<r;i++){
      for(int j =0; j<c;j++){
        input[j + i*c] = (float)rand() / RAND_MAX;
        printf("%.2f ",input[j + i*c]);
      }
      printf("\n");
    }

    cudaMemcpy(in,input,c*r*sizeof(float),cudaMemcpyHostToDevice);

    dim3 gridsize(ceil((float)r/blocksize),ceil((float)c/blocksize));
    dim3 BlockSize(blocksize,blocksize);

    matrix_tranpose<<<gridsize,BlockSize>>>(in,out);

    cudaMemcpy(output,out,r*c*sizeof(float),cudaMemcpyDeviceToHost);

    printf("visualizing the ouput matrix\n");
    for(int i =0; i<c;i++){
      for(int j =0; j<r;j++){
        printf("%.2f ",output[j+i*r]);
      }
      printf("\n");
    }

    cudaFree(in);
    cudaFree(out);
    free(input);
    free(output);
}
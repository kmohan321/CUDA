#include<stdio.h>
#include<cuda.h>

#define r 8
#define c 8
#define blocksize 4
#define k 3


__global__ void naive_conv2d(float*in,float *out, float*kernel){

      int row = threadIdx.x + blockDim.x * blockIdx.x; //rows
      int col = threadIdx.y + blockDim.y * blockIdx.y; //columns
      
      float value = 0.0f;
      if(row < (r-k+1) && col < (c-k+1)){
        for(int i =0; i < k; i++){
          for(int j =0; j < k; j++){
            value += in[col+j + (i+row)*c] * kernel[j+i*k];
          }
        }
      }
      if(row < (r-k+1) && col < (c-k+1)){
          out[col + row * (c-k+1)] = value;
      }
}

int main(){

    float *input = (float*)malloc(r * c * sizeof(float));
    float *output = (float*)malloc((r-k+1) * (c-k+1) * sizeof(float));
    float *kernel_h = (float*)malloc(k * k * sizeof(float));

    float *in, *out, *kernel;

    cudaMalloc((void**)&in , r*c*sizeof(float));
    cudaMalloc((void**)&out , (r-k+1)*(c-k+1)*sizeof(float));
    cudaMalloc((void**)&kernel , k*k*sizeof(float));

    //let's intialize the matrix
    for(int i =0; i<r;i++){
      for(int j =0; j<c;j++){
        input[j + i*c] = (float)rand() / RAND_MAX;
        printf("%.2f ",input[j + i*c]);
      }
      printf("\n");
    }

    //let's intialize the kernel
    printf("visualizing the kernel\n"); 
    for(int i =0; i<k;i++){
      for(int j =0; j<k;j++){
        kernel_h[j + i*c] = (float)rand() / RAND_MAX;
        printf("%.2f ",kernel_h[j + i*k]);
      }
      printf("\n");
    }

    cudaMemcpy(in,input,c*r*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(kernel,kernel_h,k*k*sizeof(float),cudaMemcpyHostToDevice);

    dim3 gridsize(ceil((float)r/blocksize),ceil((float)c/blocksize));
    dim3 BlockSize(blocksize,blocksize);

    naive_conv2d<<<gridsize,BlockSize>>>(in,out,kernel);

    cudaMemcpy(output,out,(r-k+1)*(c-k+1)*sizeof(float),cudaMemcpyDeviceToHost);

    printf("visualizing the ouput matrix\n");
    for(int i =0; i<(r-k+1);i++){
      for(int j =0; j<(c-k+1);j++){
        printf("%.2f ",output[j+i*(c-k+1)]);
      }
      printf("\n");
    }

    cudaFree(in);
    cudaFree(out);
    cudaFree(kernel);
    free(input);
    free(output);
    free(kernel_h);
}

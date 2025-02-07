#include<stdio.h>
#include<cuda.h>

#define r 1000
#define c 1000
#define common 1000
#define blocksize 32 //tiling same

//remember each thread calculates the one element of the output matrix

__global__ void linear_layer(float *input, float *weight, float*bias,float *output){

    __shared__ float s1[blocksize*blocksize];
    __shared__ float s2[blocksize*blocksize];

    int x = threadIdx.x / blocksize;
    int y = threadIdx.x % blocksize;

    int row = blockIdx.y;
    int col = blockIdx.x;
    
    input += row * blocksize * common;
    weight += col * blocksize;
    output += row * blocksize * c + col * blocksize; 
    bias += row * blocksize * c + col * blocksize; 

    float sum =0.0f;
    for(int blockidx =0;blockidx<common; blockidx+=blocksize){

        s1[x + y*blocksize] = input[x + y*common];
        s2[x + y*blocksize] = output[x + y*c];

        __syncthreads();

        input += blocksize;
        weight += blocksize * c;

        for (int dotIdx = 0; dotIdx < blocksize; ++dotIdx) {
            sum += s1[y * blocksize + dotIdx] *
                    s2[dotIdx * blocksize + x];
          }
        __syncthreads();
        

    }
    output[y * c + x] = sum + bias[y*c+x];

}

int main(){
      
    float *mat1 = (float*)malloc(r*common*sizeof(float));
    float *mat2 = (float*)malloc(common*c*sizeof(float));
    float *mat3 = (float*)malloc(r*c*sizeof(float));
    float *mat4 = (float*)malloc(r*c*sizeof(float));

    float *m1, *m2, *m3, *m4;
    cudaMalloc((void**)&m1, r*common*sizeof(float));
    cudaMalloc((void**)&m2, common*c*sizeof(float));
    cudaMalloc((void**)&m3, r*c*sizeof(float));
    cudaMalloc((void**)&m4, r*c*sizeof(float));

    for(int i =0; i<r; i++){   //shape -> //(r,common)
          for(int j =0; j<common; j++){
                mat1[j + common *i] = i + 1;
                }
    }

    for(int i =0; i<common; i++){  //shape -> //(common,c)
          for(int j =0; j<c; j++){
                mat2[j + c *i] = i*2;
          }
    }

    for(int i =0; i<common; i++){  //shape -> //(r,c)
          for(int j =0; j<c; j++){
                mat4[j + c *i] = i*2;
          }
    }
    cudaMemcpy(m1,mat1,r*common*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(m2,mat2,common*c*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(m4,mat4,r*c*sizeof(float),cudaMemcpyHostToDevice);

    dim3 blocks(ceil((float)r/blocksize),ceil((float)c/blocksize));
    dim3 block_size(blocksize*blocksize);
    
    //kernel time
    linear_layer<<<blocks , block_size>>>(m1,m2,m4,m3);
    cudaMemcpy(mat3,m3,r*c*sizeof(int),cudaMemcpyDeviceToHost);

    cudaFree(m1);
    cudaFree(m2);
    cudaFree(m3);
    free(mat1);
    free(mat2);
    free(mat3);

    return 0;
}
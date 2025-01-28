#include<stdio.h>
#include<cuda.h>

#define r 10000
#define c 10000
#define m 16

__global__ void matrix_add(int *m1, int *m2, int *m3){
      int idx = blockDim.x * blockIdx.x + threadIdx.x; //columns for  matrix
      int idy = blockDim.y * blockIdx.y + threadIdx.y; //rows for matrix 
      if(idx<c && idy <r){
           m3[idx + idy * c ] = m1[idx + idy *c] + m2[idx + idy * c];
      }
}

int main(){
  
  //let's intialize the matrices
  int *mat1 = (int*)malloc(r*c*sizeof(int));
  int *mat2 = (int*)malloc(r*c*sizeof(int));
  int *mat3 = (int*)malloc(r*c*sizeof(int));

  int *m1, int *m2, int*m3;
 
  // pointers to the memory location on gpus
  cudaMalloc((void**)&m1, r*c*sizeof(int));
  cudaMalloc((void**)&m2, r*c*sizeof(int));
  cudaMalloc((void**)&m3, r*c*sizeof(int));
  
  for(int i =0; i<r; i++){
    for(int j =0; j<c; j++){
      mat1[i + c *j] = i + 1;
      mat2[i + c *j] = i*2;
    }
  }

  // visulazing the matrices
  printf("matrix1\n");
  for(int i =0; i<6; i++){
    for(int j =0; j<6; j++){
      printf("%d ", mat1[i + c*j]);
    }
    printf("\n");
  }
  printf("matrix2\n");
  for(int i =0; i<6; i++){
    for(int j =0; j<6; j++){
      printf("%d ", mat2[i + c* j]);
    }
    printf("\n");
  }

  cudaMemcpy(m1,mat1,r*c*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(m2,mat2,r*c*sizeof(int),cudaMemcpyHostToDevice);
  
  //launching the cuda kernel 
  int blocks_x =  (c + m -1) / m;
  int blocks_y =  (r + m -1) / m;

  dim3 grid(blocks_x,blocks_y);
  dim3 blocksize(m,m);
  
  matrix_add<<<grid,blocksize>>>(m1,m2,m3);

  cudaMemcpy(mat3,m3,r*c*sizeof(int),cudaMemcpyDeviceToHost);
  printf("output matrix\n");
  for(int i =0; i<6; i++){
    for(int j =0; j<6; j++){
      printf("%d ", mat3[i + c*j]);
    }
    printf("\n");
  }

  // Free memory
  free(mat1);
  free(mat2);
  free(mat3);
  cudaFree(m1);
  cudaFree(m2);
  cudaFree(m3);

  return 0;
}



#include<stdio.h>
#include<cuda.h>

#define r 1000
#define c 1000
#define common 500
#define threads 16 //threads per dimension of block

__global__ void matrix_mul(int *m1, int *m2, int *m3){

      int idx = threadIdx.x + blockDim.x * blockIdx.x; //column index
      int idy = threadIdx.y + blockDim.y * blockIdx.y; //row index

      if(idx<c && idy <r){
            int sum = 0;
            for(int i =0; i<common; i++){
                  sum += m1[i + idy*common] * m2[i*c + idx];
            }
            m3[idx + idy * c] = sum;
      }
}

int main(){
      
      int *mat1 = (int*)malloc(r*common*sizeof(int));
      int *mat2 = (int*)malloc(common*c*sizeof(int));
      int *mat3 = (int*)malloc(r*c*sizeof(int));

      int *m1, *m2, *m3;
      cudaMalloc((void**)&m1, r*common*sizeof(int));
      cudaMalloc((void**)&m2, common*c*sizeof(int));
      cudaMalloc((void**)&m3, r*c*sizeof(int));

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
      cudaMemcpy(m1,mat1,r*common*sizeof(int),cudaMemcpyHostToDevice);
      cudaMemcpy(m2,mat2,common*c*sizeof(int),cudaMemcpyHostToDevice);

      dim3 blocks(ceil(c/threads),ceil(r/threads)); //out matrix shape-> (1000,1000) thredas-> 1000*1000
      // each block size(256,256) blocks->(c/256,r/256)
      dim3 block_size(threads,threads);
      
      //kernel time
      matrix_mul<<<blocks , block_size>>>(m1,m2,m3);
      cudaMemcpy(mat3,m3,r*c*sizeof(int),cudaMemcpyDeviceToHost);

      //visualizing the output
      printf("final matrix.. \n");
      for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                  printf("%d ", mat3[i * c + j]); 
            }
            printf("\n"); 
      }

      cudaFree(m1);
      cudaFree(m2);
      cudaFree(m3);
      free(mat1);
      free(mat2);
      free(mat3);

      return 0;
}
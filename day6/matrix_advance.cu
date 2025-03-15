#include<stdio.h>
#include<cuda.h>

#define r 3
#define c 3
#define common 4
#define threads 2 //threads per dimension of block
#define tile 2

__global__ void matrix_mul(int *m1, int *m2, int *m3){

      int idx = threadIdx.x + blockDim.x * blockIdx.x; //column index
      int idy = threadIdx.y + blockDim.y * blockIdx.y; //row index
      int x = threadIdx.x;
      int y = threadIdx.y;
      
      __shared__ int s1[tile*tile];
      __shared__ int s2[tile*tile];

      int sum = 0; 
      for(int j =0;j<(common + tile -1)/tile;j++){

            if(idy <r && idx <common){
                  s1[x + y*tile] = m1[x + idy * common + j*tile];
            }
            __syncthreads();

            if(idx<c && idy<common){
                  s2[x + y*tile] = m2[idx + y *c + c*tile*j]; 
            }
            __syncthreads();

            // if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==1 && blockIdx.y==1){
            // for(int i =0; i<tile;i++){
            //       for(int j=0;j<tile;j++){
            //             printf("%d ",s2[j + i*tile]);
            //       }
            //       printf("\n");
            // }  }
            
            if(idx<c && idy<r){
                  for(int i =0 ; i<tile;i++){
                        sum += s1[i + y*tile] * s2[x + i*tile];
                  }
            }
            __syncthreads();
      }

      if(idx<c && idy<r){
            m3[idx + idy*c] =sum;
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

      dim3 blocks(ceil((float)c/threads),ceil((float)r/threads)); //out matrix shape-> (1000,1000) thredas-> 1000*1000
      // each block size(256,256) blocks->(c/256,r/256)
      dim3 block_size(threads,threads);
      
      //kernel time
      matrix_mul<<<blocks , block_size>>>(m1,m2,m3);
      cudaMemcpy(mat3,m3,r*c*sizeof(int),cudaMemcpyDeviceToHost);

      // visualizing the output
      printf("final matrix.. \n");
      for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
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
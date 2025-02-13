#include<stdio.h>
#include<cuda.h>

#define blocksize 32
#define r 16
#define c 16
#define radius 1 

__global__ void game_kernel (float *input, float *output){

    __shared__ float smem[blocksize*blocksize];

    int x = threadIdx.x;
    int y = threadIdx.y;

    int col = x + blockDim.x * blockIdx.x;
    int row = y + blockDim.y * blockIdx.y;

    //loading the elements in the shared memory
    if(row<r && col <c){
      smem[x + y*blocksize] = input[col + row*c];
    }
    else{
      smem[x + y*blocksize] = 0.0f;
    }
    __syncthreads();
    
    int local_x = x - radius; //starting point for x
    int local_y = y - radius; //starting point for y

    float alive = 0;
    if(row<r && col<c){
      for(int n_row = 0; n_row<2*radius+1;n_row++){
        for(int n_col =0; n_col<2*radius+1;n_col++){
            if((local_x+n_col) >= 0 && (local_y+n_row)>=0 && (local_x+n_col)<blocksize && (local_y+n_row)<blocksize){
              if((local_x+n_col) == local_x && (local_y+n_row)==local_y) continue;    
              if(smem[(local_x + n_col) + (local_y+n_row) * blocksize]==1){
                    alive += 1;
                  }
            }
        }
      }
      if(smem[x+y*blocksize]==1&& alive<2){
        output[col + row*c] = 0; //underpopulation
      }
      else if(smem[x+y*blocksize]==1&& alive>=2 && alive<3){
        output[col + row*c] = 1; 
      }
      else if(smem[x+y*blocksize]==1&& alive>3){
        output[col + row*c] = 0;  //overpopulation
      }
      else if (smem[x + y * blocksize] == 0 && alive == 3) {
        output[col + row * c] = 1;  // Reproduction 
      } else {
        output[col + row * c] = smem[x + y * blocksize];  
      }
    }
}


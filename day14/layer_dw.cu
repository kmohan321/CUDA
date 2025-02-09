#include<stdio.h>
#include<cuda.h>

#define r 1000
#define c 1000
#define k 500
#define blocksize 32

/* here dw -> gradient of loss wrt to weight shape->(Din,Dout)
 and X_t transposed input shape->(Din,B) also known as gradient of output with respect to weight
 and dy  gradient of loss with respect to the ouput shape->(B,Dout)
*/

// kernel for dw
__global__ void dW (float*dW, float*X_t, float*dy){

  int col = threadIdx.x + blockIdx.x * blockDim.x; //globalising the threads
  int row = threadIdx.y + blockIdx.y * blockDim.y;

  __shared__ float s1[blocksize*blocksize];
  __shared__ float s2[blocksize*blocksize];

  int x = threadIdx.x;
  int y = threadIdx.y;  
  float sum = 0.0f;
  //let's roll over the tiles
  for(int tileid = 0 ;tileid < ceil((float)k/blocksize); tileid++){

      //filling shared memory from X_t 
      if(row<r && (x + tileid*blocksize)<k){
        s1[x + y*blocksize] = X_t[x + row*k + tileid*blocksize];
      }
      else{
        s1[x + y*blocksize] = 0.0f;
      }
      
      //filling shared memory from dy
      if((y + blocksize*tileid)<k && col <c){
        s2[x + y*blocksize] = dy[col + y*c + c*(tileid*blocksize)];
      }
      else{
        s2[x + y*blocksize] = 0.0f;
      }
      __syncthreads();

      //computing the partial sum
      for(int sid = 0; sid<blocksize;sid++){
        sum += s1[sid + blocksize*y] * s2[x + sid*blocksize];
      }
      __syncthreads();
  }
  if(row<r && col<c){
    dW[col + row * c] = sum;
  }
}

//kernel db ->gradient with respect to the bias
// each block will handle the each row 
// shape of dy (r,c) db -> shape (r)
__global__ void dx (float*db, float*dy){

  int x = threadIdx.x;

  __shared__ float s1[blocksize*blocksize];

  int row = blockIdx.x;
  
  float sum =0.0f; //local sum
  for(int stride =x; x<c; stride+= blockDim.x){
      sum += dy[x + row*c];
  }
  s1[x] = sum;
  __syncthreads();

  //reduction time
  for(int i = blocksize/2;i>0;i/=2){
    if(x<i){
      s1[x]+= s1[x+i];
    }
    __syncthreads();
  }
  float global_sum = s1[0];
  db[row] = global_sum;
}


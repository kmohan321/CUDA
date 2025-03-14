#include <stdio.h>
#include <cuda.h>


__global__ void matrix_mul(float*m1, float*m2 , float*m3,int height, int width,int common){
    int idx = threadIdx.x + blockDim.x * blockIdx.x; // columns for result matrix
    int idy = threadIdx.y + blockDim.y * blockIdx.y; // rows for result matrix

    if(idx<width && idy <height){
      float value = 0;
      for(int i=0; i<common;i++){
          value += m1[i + idy*width] * m2[i * width + idx ];
      }
      m3[idx + idy * width] = value;
    }

}


int main(){

  float array1[3][4] = {
        {1.1, 2.2, 3.3, 4.4},
        {5.5, 6.6, 7.7, 8.8},
        {9.9, 10.1, 11.2, 12.3}
  };
  float *m1_h = *array1;

  float array2[4][3] = {
    {1,3,4},
    {3,6,7},
    {5,7,9},
    {4,9,0}
  };

  float *m2_h = *array2;
  
  float *m1_d, *m2_d , *m3_d, *m3_h; 
  cudaMalloc(&m1_d,sizeof(float)*12);
  cudaMalloc(&m2_d,sizeof(float)*12);
  cudaMalloc(&m3_d,sizeof(float)*9);

  m3_h = (float*)malloc(sizeof(float)*9);

  cudaMemcpy(m2_d,m2_h,sizeof(float)*12,cudaMemcpyHostToDevice);
  cudaMemcpy(m1_d,m1_h,sizeof(float)*12,cudaMemcpyHostToDevice);
  dim3 threads(3,3);
  dim3 blocks(1,1);
  matrix_mul<<<blocks,threads>>>(m1_d,m2_d,m3_d,3,3,4);
  cudaDeviceSynchronize();
  cudaMemcpy(m3_h,m3_d,sizeof(float)*9,cudaMemcpyDeviceToHost);

  //printf("sucess");
  for(int i =0; i<3;i++){
    for (int j=0; j<3 ;j++){
      printf("%.2f ",m3_h[j + i *3]);
      if (j!=0 && j==2){
        printf("\n");
      }
    }
  }
  return 0;
}
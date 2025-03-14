#include<stdio.h>
#include<cuda_runtime.h>

__global__ void kernel(float *in_img , float *out_img , int int_width, int int_height){

 int id_x = threadIdx.x + blockIdx.x * blockDim.x;
 int id_y = threadIdx.y + blockIdx.y * blockDim.y;

 if(id_x<int_width && id_y<int_height){

    int id = (id_x + id_y * int_width)*3;
    float value_r  = 0.21 * in_img[id];
    float value_g = 0.72 * in_img[id + 1] ;
    float value_b = 0.07 * in_img[id + 2];

    out_img[id_x + id_y * int_width] =  value_r + value_g + value_b;
 }

 }


extern "C"{
__declspec(dllexport) void main(float * in_img , float *out_img , int int_width, int int_height){
    float * img_d , *img_h;
    cudaMalloc(&img_d, sizeof(float)*int_width*int_height*3);
    cudaMalloc(&img_h, sizeof(float)*int_width*int_height);
    cudaMemcpy(img_d, in_img, sizeof(float)*int_width*int_height*3, cudaMemcpyHostToDevice);

    dim3 blocks(ceil(int_height/32),ceil(int_width/32),1);
    dim3 threads(32,32,1);
    kernel<<<blocks,threads>>>(img_d,img_h,int_height,int_width);

    cudaMemcpy(out_img, img_h, sizeof(float)*int_width*int_height, cudaMemcpyDeviceToHost);
}
}


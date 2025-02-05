#include<stdio.h>
#include<cuda.h>

#define blocksize 1024
#define r 1024
#define c 32768

/*we will use each block to process entire row */
__global__ void soft_opt(float *input, float *output){

    __shared__ float s[blocksize];
    
    int x = threadIdx.x;
    int row = blockIdx.x; //for rows
    
    float norm = 0.0f;
    float max_value = -INFINITY;
    //benefit each thread can access the elements as much as it needed for the row (no restriction for column size)
    for(int i = x; i<c; i += blockDim.x){ //reason for doing this is because of coalesing 

        float current_value = input[i + row*c];
        if(current_value > max_value){
            norm = norm * expf(max_value-current_value);
            max_value = current_value;
        }
        norm += expf(current_value-max_value);
    }
    s[x] = max_value;
    __syncthreads();
    
    //reduction time
    for(int stride = blockDim.x/2 ; stride>0; stride /=2 ){
        if(x<stride){
            s[x] = fmax(s[x],s[x+stride]);
        }
    }
    __syncthreads();
    float global_maxvalue = s[0];

    //so now we have global_maxvalue, time for corrected norm
    norm *= expf(max_value-global_maxvalue);
    s[x] = norm;
    __syncthreads();

    //reduction time for norm
    for(int stride = blockDim.x/2 ; stride>0; stride /=2 ){
        if(x < stride){
            s[x] += s[x+stride];
        }
    }
    __syncthreads();
    float final_norm = s[0];

    //time for softmax
    for(int i = x; i<c; i += blockDim.x){
        output[x + row*c] = expf(input[i + row*c]-global_maxvalue) / final_norm;
    }
    
}

int main(){

    float *input = (float*)malloc(r * c * sizeof(float));
    float *output = (float*)malloc(r * c * sizeof(float));

    float *in, *out;

    cudaMalloc((void**)&in ,r*c*sizeof(float));
    cudaMalloc((void**)&out , r*c*sizeof(float));

    //let's intialize the matrix
    for(int i =0; i<r;i++){
      for(int j =0; j<c;j++){
        input[j + i*c] = (float)rand() / RAND_MAX;
      }
    }

    cudaMemcpy(in,input,r*c*sizeof(float),cudaMemcpyHostToDevice);

    dim3 gridsize(r);
    dim3 Block_Size(blocksize);

    soft_opt<<<gridsize,Block_Size>>>(in,out);

    cudaMemcpy(output,out,r*c*sizeof(float),cudaMemcpyDeviceToHost);

    // for debugging 
    // for(int i =0; i<r;i++){
    //   float sum = 0.0f;
    //   for(int j =0; j<c;j++){
    //     sum += output[j+i*c];
    //   }
    //   printf("\nsum is %.2f \n",sum);
    // }

    cudaFree(in);
    cudaFree(out);
    free(input);
    free(output);

}
#include<stdio.h>
#include<cuda.h>

#define blocksize 32
#define k 500
#define r 1000
#define c 1000 //will set as parameter during full neural net buid

__global__ void dx (float*dx, float*dy, float*W_t){

    int col = threadIdx.x + blockIdx.x * blockDim.x; //globalising the threads
    int row = threadIdx.y + blockIdx.y * blockDim.y;

    __shared__ float s1[blocksize*blocksize];
    __shared__ float s2[blocksize*blocksize];

    int x = threadIdx.x;
    int y = threadIdx.y;  
    float sum = 0.0f;
    //let's roll over the tiles
    for(int tileid = 0 ;tileid < ceil((float)k/blocksize); tileid++){

        //filling shared memory from dy 
        if(row<r && (x + tileid*blocksize)<k){
          s1[x + y*blocksize] = dy[x + row*k + tileid*blocksize];
        }
        else{
          s1[x + y*blocksize] = 0.0f;
        }
        
        //filling shared memory from W_t
        if((y + blocksize*tileid)<k && col <c){
          s2[x + y*blocksize] = W_t[col + y*c + c*(tileid*blocksize)];
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
      dx[col + row * c] = sum;
    }
}


// CPU fucntion for validation
void cpu_matrix_multiply(float* dx, float* dy, float* W_t) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            float sum = 0.0f;
            for (int m = 0; m < k; m++) {
                sum += dy[m + i * k] * W_t[j + m * c];
            }
            dx[j + i * c] = sum;
        }
    }
}

//comparison function
bool compare_results(float* cpu_result, float* gpu_result, int size) {
    const float epsilon = 1e-2;
    for (int i = 0; i < size; i++) {
        if (fabsf(cpu_result[i] - gpu_result[i]) > epsilon) {
            printf("Mismatch at index %d: CPU = %f, GPU = %f\n", 
                   i, cpu_result[i], gpu_result[i]);
            return false;
        }
    }
    return true;
}

int main() {
    float *h_dx, *h_dy, *h_W_t;
    float *h_dx_cpu; 
    
    h_dx = (float*)malloc(r * c * sizeof(float));
    h_dy = (float*)malloc(r * k * sizeof(float));
    h_W_t = (float*)malloc(k * c * sizeof(float));
    h_dx_cpu = (float*)malloc(r * c * sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < r * k; i++) {
        h_dy[i] = (float)(rand()) / RAND_MAX;
    }
    for (int i = 0; i < k * c; i++) {
        h_W_t[i] = (float)(rand()) / RAND_MAX;
    }

    float *d_dx, *d_dy, *d_W_t;
    cudaMalloc(&d_dx, r * c * sizeof(float));
    cudaMalloc(&d_dy, r * k * sizeof(float));
    cudaMalloc(&d_W_t, k * c * sizeof(float));

    cudaMemcpy(d_dy, h_dy, r * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_t, h_W_t, k * c * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 threadsPerBlock(blocksize, blocksize);
    dim3 numBlocks((c + blocksize - 1) / blocksize, 
                   (r + blocksize - 1) / blocksize);

    cudaEventRecord(start);
    dx<<<numBlocks, threadsPerBlock>>>(d_dx, d_dy, d_W_t);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_milliseconds = 0;
    cudaEventElapsedTime(&gpu_milliseconds, start, stop);

    cudaMemcpy(h_dx, d_dx, r * c * sizeof(float), cudaMemcpyDeviceToHost);

    clock_t cpu_start = clock();
    cpu_matrix_multiply(h_dx_cpu, h_dy, h_W_t);
    clock_t cpu_end = clock();
    double cpu_milliseconds = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;

    bool results_match = compare_results(h_dx_cpu, h_dx, r * c);

    printf("GPU Time: %f ms\n", gpu_milliseconds);
    printf("CPU Time: %f ms\n", cpu_milliseconds);
    printf("Speedup: %.2fx\n", cpu_milliseconds / gpu_milliseconds);
    printf("Results match: %s\n", results_match ? "Yes" : "No");

    cudaFree(d_dx);
    cudaFree(d_dy);
    cudaFree(d_W_t);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(h_dx);
    free(h_dy);
    free(h_W_t);
    free(h_dx_cpu);

    return 0;
}




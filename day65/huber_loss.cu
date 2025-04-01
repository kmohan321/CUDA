#include<stdio.h>
#include<cuda.h>

__global__ void huber_loss(float *logits, float *labels, float *loss, int M, int N, float delta){

    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    
    if(row < M && col < N){

        float curr_logits = logits[row * N + col];
        float curr_labels = labels[row *N + col];

        float abs_distance = fabsf(curr_logits - curr_labels);
        if(abs_distance<delta){
            loss[row*N + col] = 0.5f * abs_distance * abs_distance;
        }
        else{
            loss[row * N + col] = delta * (abs_distance - 0.5f * delta);
        }
    }
}




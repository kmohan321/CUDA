#include<stdio.h>
#include<cuda.h>

#define r 64
#define c 100
#define d 128

__global__ void Attention(float *Q, float *K, float *V, float *O, int seq_len, int dim)
{
    int row = threadIdx.y + blockIdx.y * blockDim.y;  // Sequence index
    int col = threadIdx.x + blockIdx.x * blockDim.x;  // Feature index

    extern __shared__ float attn_scores[];  

    float softmaxscale = 1.0f / sqrtf(dim);

    if (row < seq_len && col < dim)
    {
        
        float sum = 0.0f;
        for (int k = 0; k < dim; k++)
        {
            sum += Q[row * dim + k] * K[k * dim + col];  
        }

        attn_scores[row * dim + col] = sum * softmaxscale;
        __syncthreads();

      
        float max_value = -1e30f;
        for (int k = 0; k < dim; k++)
        {
            max_value = fmaxf(max_value, attn_scores[row * dim + k]);
        }
        __syncthreads();


        float denum = 0.0f;
        for (int k = 0; k < dim; k++)
        {
            attn_scores[row * dim + k] = expf(attn_scores[row * dim + k] - max_value);
            denum += attn_scores[row * dim + k];
        }
        __syncthreads();

        for (int k = 0; k < dim; k++)
        {
            attn_scores[row * dim + k] /= denum;
        }
        __syncthreads();

        sum = 0.0f;
        for (int k = 0; k < seq_len; k++)  
        {
            sum += attn_scores[row * dim + k] * V[k * dim + col];  
        }

        O[row * dim + col] = sum;
    }
}






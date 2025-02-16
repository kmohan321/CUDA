#include<stdio.h>
#include<cuda.h>


//rememeber inputs are 4d tensors -> (b,h,s,d)
__global__ void flash_attn(float *K, float *V, float *Q, float *l, float *m,
    int T_r, int T_c, int b, int h, int s, int d,int B_c, int B_r){

    //memory allocation  
    extern __shared__ float smem[];

    float *k = smem;
    float *v = smem + B_c * d;
    float *q = smem + 2 * B_c * d;
    float *o = smem + 2* B_c *d + B_r*d;

    //indexing 
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int x = threadIdx.x; // for B_r elements (total elements in each query tile)
    int y = threadIdx.y; // for B_c elements (total elements in each key and value tile)

    //loading the values
    float l_i = 0.0f;
    float m_i = 0.0f;
    float S[B_c];
    
    for(int j = 0 ; j < T_c; j++){

        //loading the K and V tiles
        for (int c = 0; c < d ; c++){

            k[y*d + c] = K[batch_idx * (h*s*d) + head_idx * (s*d) + (B_c *j*d + y*d) + c];
            v[y*d + c] = V[batch_idx * (h*s*d) + head_idx * (s*d) + (B_c *j*d + y*d) + c];
        }
        __syncthreads();

        for(int i = 0; i < T_r; i++){
            
            //loading the Q tiles
            for(int c = 0; c<d ; c++){
                q[x*d + c]  = Q[batch_idx * (h*s*d) + head_idx * (s*d) + (B_r *i*d + x*d) + c];
            }
            __syncthreads();
            
            //local values
            l_i = l[B_r * i + y]; 
            m_i = m[B_r * i + y];

            //computing the dot product 
            for(int l_row = 0; l_row < B_c; l_row++){
                float S_ij = 0.0f;
                for(int common = 0 ; common<d; common++){
                    S_ij +=  q[x*d + common] * k[l_row*d + common];
                }
                s[l_row] = S_ij;
            }

            //performing the softmax
            for(int l_row =0; l_row<B_c; l_row++){
                float curr_value = s[l_row];
                if(curr_value > m_i){
                    l_i = l_i * expf(m_i - curr_value);
                    m_i = curr_value;
                }
                l_i += expf(curr_value-m_i);
            }

            //computing the final output
            for(int l_row =0; l_row < d; l_row++){
                float local_sum = 0.0f;
                for(int common =0; common < B_c; common++){
                    local_sum += S[common] * v [common*d + d];
                }
            
            
            //since you have a query 0 on thread 0 having B_c attention weights
            //and have B_c values having d dimension but so you will have partial sum for d0 
            //position for query 0 
            //save them in O in d postion ok 

            }

        }
    }
}



//grid size (b,num_heads)
//blocksize () pata nahi 
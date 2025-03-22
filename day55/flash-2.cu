#include<stdio.h>
#include<cuda.h>

__global__ void flash_attn2(float *Query, float *Key, float *Value,float *Output, int s,
  int d, int heads, int Br, int Tc, int Bc,float scale){

    extern __shared__ float smem[];

    int Br_idx = blockIdx.x; //parallelizing the blocks of query(T_r)
    int batch_head = blockIdx.y; //batch_head pair
    int idx = threadIdx.x; //for each query in block

    int head_idx = batch_head % heads;
    int batch_idx = batch_head / heads;

    float *q_block = smem;
    float *k_block = q_block + Br * d;
    float *v_block = k_block + Bc * d;
    float *o_block = v_block + Bc *d;
    float *attn_scores = o_block + Br*d;

    int batch_head_offset = batch_idx * s*heads*d + head_idx * s *d; //move to the correct batch and head

    //loading the query block
    if(Br_idx + idx < s){
    for(int i = 0; i<d; i++){
        q_block[idx * d + i] = Query[batch_head_offset + (Br_idx + idx)*d + i]; //Br*d
      }
    }
    __syncthreads();
    
    //looping over the key and value blocks
    //j -> block_idx for key and values bc_idx

    float prev_local_max = -INFINITY , local_sum = 0.0f; //local_sum will be used after this loop
    for(int bc_idx =0; bc_idx < Tc; bc_idx++){

      //loading the key and value tiles
      if((bc_idx + idx) < s){
        for(int i =0 ; i<d ;i++){
          k_block[idx *d + i] = Key[batch_head_offset + (bc_idx + idx)*d + i]; //Bc*d
        }  
      }
      __syncthreads();

      if((bc_idx + idx) < s){
        for(int i =0 ; i<d ;i++){
          v_block[idx *d + i] = Value[batch_head_offset + (bc_idx + idx)*d + i]; //Bc*d
        }  
      }
      __syncthreads();

      //taking the dot product 
      //Br* d @ d *Bc -> Br* Bc
      for(int i = 0; i < Bc ; i++){
        float accumulator = 0.0f;
        for(int common =0 ; common <d ; common++){
          accumulator += q_block[idx * d + common] * k_block[i * d + common];
        }
        attn_scores[idx * Bc + i] = accumulator;
      }
      
      //rowwise max and sum adjustment
      float local_max = -INFINITY;
      for(int i = 0; i < Bc; i++){
        float curr_value = attn_scores[idx * Bc + i];
        if(curr_value >local_max){
          local_sum = local_sum * expf(local_max - curr_value);
          local_max = curr_value;
        }
        local_sum += expf(curr_value-local_max);
      }

      //load the Ouput into shared memory
      if(Br_idx + idx <s){
        for(int i = 0 ; i<d ; i++){
          o_block[idx * d + i] = Output[batch_head_offset + (Br_idx + idx)*d + i]; //Br*d
        }
      }
      __syncthreads();

      //final product with the v block
      //Br* Bc @ Bc *d -> Br* d
      float correction_factor = expf(prev_local_max - local_max);
      for(int i = 0; i<d; i++){
        float curr_value = o_block[idx * d + i] * correction_factor;
        for(int common = 0; common<Bc; common++){
          o_block[idx * d + i] = curr_value + expf(attn_scores[idx * Bc + common] - local_max) * v_block[common *d + i];
        }
      }
      prev_local_max = local_max;
    }

    for(int i = 0 ; i<d; i++){
      Output[batch_head_offset + idx *d + i] = o_block[idx * d+ i] / local_sum;
    }

}


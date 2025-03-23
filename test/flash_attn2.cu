#include<stdio.h>
#include<cuda.h>
#include<torch/types.h>

__global__ void flash_attn2(float *Query, float *Key, float *Value,float *Output,
   int s,int d, int heads, int Br, int Tc, int Bc,float scale){

    extern __shared__ float smem[];

    int Br_idx = blockIdx.x; //parallelizing the blocks of query(T_r)
    int batch_head = blockIdx.y; //batch_head pair
    int idx = threadIdx.x; //for each query in a block

    int head_idx = batch_head % heads;
    int batch_idx = batch_head / heads;

    float *q_block = smem;
    float *k_block = q_block + Br * d;
    float *v_block = k_block + Bc * d;
    float *o_block = v_block + Bc *d;
    float *attn_scores = o_block + Br*d;

    float *row_max = attn_scores + Br*Bc;
    float *row_sum = row_max + Br;
    float *new_max = row_sum + Br;
    
    int batch_head_offset = batch_idx * s*heads*d + head_idx * s *d; //move to the correct batch and head

    //loading the query block
    int block_row = idx / Bc; //Br*Bc threads
    int block_col = idx % Bc;

    for(int i = block_col ; i < d; i += Bc){
      if(Br_idx * Br + block_row < s){
        q_block[block_row * d + i] = Query[batch_head_offset + (Br_idx *Br + block_row) * d + i];
        o_block[block_row * d + i] = 0.0f;
      }
    }
    __syncthreads();

    //looping over the key and value blocks
    //j -> block_idx for key and values bc_idx

    if(block_col==0){
      row_max[block_row] = -INFINITY;
      row_sum[block_row] = 0.0f;
      new_max[block_row] = -INFINITY; 
    }
    __syncthreads();

    for(int Bc_idx = 0; Bc_idx < Tc; Bc_idx++){

      //loading the key and value tiles
      int local_row = idx / Br;
      int local_col = idx % Br;

      for(int i = local_col ; i < d; i += Br){
        if(Bc_idx * Bc + local_row < s){
          k_block[local_row * d + i] = Key[batch_head_offset + (Bc_idx *Bc + local_row) * d + i];
          v_block[local_row * d + i] = Value[batch_head_offset + (Bc_idx *Bc + local_row) * d + i];
        }
      }     
      __syncthreads();

      //taking the dot product 
      //Br* d (dot) Bc*d -> Br* Bc
      float sum = 0.0f;
      for(int i = 0; i <d; i++){
          sum += q_block[block_row * d + i] * k_block[block_col * d + i];
      }
      attn_scores[block_row * Bc + block_col] = sum * scale;
      __syncthreads();
      
      //rowwise max and sum adjustment
      if(block_col==0){
        float local_max = -INFINITY;
        float local_sum = 0.0f;
        for(int i = 0; i < Bc; i++){
          float curr_value = attn_scores[block_row * Bc + i];
          if(curr_value > local_max){
            local_sum = local_sum * expf(local_max - curr_value);
            local_max = curr_value;
          }
          local_sum += expf(curr_value-local_max);
        }
        new_max[block_row] = max(row_max[block_row],local_max);
        row_sum[block_row] = row_sum[block_row] * expf(row_max[block_row] - new_max[block_row]) + local_sum *expf(local_max-new_max[block_row]);
      }
      __syncthreads();

      //final product with the v block
      //Br* Bc @ Bc *d -> Br* d
      for(int col = block_col; col<d; col+=Bc){
        float sum =0.0f;
        for(int k = 0 ; k<Bc; k++){
          sum += expf(attn_scores[block_row * Bc + k] - new_max[block_row]) * v_block[col + k*d];
        }
        o_block[block_row * d + col] = o_block[block_row * d + col] * expf(row_max[block_row]-new_max[block_row]) + sum;
      }
      
      if(block_col==0){
        row_max[block_row] = new_max[block_row];
      }
      __syncthreads();
    }
    for(int i = block_col ; i < d; i+=Bc){
      if((Br_idx*Br+ block_row)<s){
        Output[batch_head_offset + (Br_idx*Br + block_row)*d + i] = o_block[block_row * d+ i] / row_sum[block_row];
      }
    }
}

torch::Tensor flash_attn2_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V ){

      int Br = 16 ,Bc = 16;

      int b = Q.size(0);
      int nh = Q.size(1);
      int s = Q.size(2);
      int d = Q.size(3);

      float scale = 1.0f / sqrtf(static_cast<float>(d)); 

      auto O = torch::zeros_like(Q);

      int smem_size = (2* Br*d + 2* Bc*d + Br*Bc + 3*Br )* sizeof(float);
      int Tr = (s + Br -1)/Br;
      int Tc = (s + Bc -1)/Bc;

      dim3 BlockSize(Br*Bc);
      dim3 grid(Tr, b*nh);

      flash_attn2<<<grid, BlockSize, smem_size>>>(
        Q.data_ptr<float>(),K.data_ptr<float>(), V.data_ptr<float>(),O.data_ptr<float>(),
        s, d, nh, Br, Tc, Bc,scale
      );

      return O;

} 


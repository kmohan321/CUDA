#include<stdio.h>
#include<cuda.h>
// #include<torch/types.h>


//matrix_mul kernel
__global__ void matrix_mul_bias(float *X, float *Y, float *bias, float *XY, int M, int N, int K, int tile_s){

    /*
      X : fisrt matrix => shape(M,K)
      Y : second matrix => shape(K,N)
      XY : output matrix => shape (M,N)
    */
   extern __shared__ float smem[];
   float *X_tile = smem;
   float *Y_tile = smem + tile_s * tile_s;

    int x  = threadIdx.x;
    int y = threadIdx.y;
    int row = y + blockDim.y * blockIdx.y;
    int col = x + blockDim.x * blockIdx.x;


    float sum = 0.0f;  
    for(int tile = 0; tile < (K + tile_s -1 )/ tile_s; tile++){

      //loading the matrices into shared memory 
      if(row < M && (x + tile*tile_s)<K){
        X_tile[y * tile_s + x] = X[ row * K + x + tile * tile_s]; 
      }
      else{
        X_tile[y * tile_s + x] = 0.0f;
      }

      if(col < N && (y + tile * tile_s)<K){
        Y_tile[y * tile_s + x] = Y[col + y * N + tile * tile_s * N ];
      }
      else{
        Y_tile[y * tile_s + x] = 0.0f;
      }
      __syncthreads();

      for(int k = 0 ; k < tile_s; k ++){
          sum += X_tile[y *tile_s + k] * Y_tile[k * tile_s + x];
      }
      __syncthreads();
   }
   if(row < M && col < N){
    XY[row * N + col] = sum + bias[col];
   }
}

//kernel for fused softmax crossentropy 
__global__ void fused_crossentropy(float *logits, int *labels, float *loss, int chunksize, int N,int blocksize){

  extern __shared__ float smem[];  

    int row = blockIdx.x;
    if(row >= chunksize) return;
    int x = threadIdx.x;

    //performing the online softmax
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    for(int i = x ; i < N; i += blocksize){
      float curr_value  = logits[row * N + i];
      if(curr_value > local_max){
        local_sum = local_sum * expf(local_max - curr_value);
        local_max = curr_value;
      }
      local_sum += expf(curr_value - local_max);
    }

    smem[x] = local_max;
    __syncthreads();
    //reduction time
    for(int i = blockDim.x/2 ; i>0 ; i /=2){
      if(x<i){
        smem[x] = fmaxf(smem[x], smem[x+i]);
      }
      __syncthreads();
    }

    float row_max = smem[0];
    local_sum = local_sum * expf(local_max - row_max);
    smem[x] = local_sum;
    __syncthreads();
    //reduction time
    for(int i = blockDim.x/2 ; i >0 ; i/=2){
      if(x<i){
        smem[x] += smem[x+i];
      }
      __syncthreads();
    }

    float row_norm = smem[0];

    int label_id = labels[row];
    for(int i = x ; i < N; i += blocksize){
        float curr_value  = logits[row * N + i];
        float y_hat = expf(curr_value - row_max) / row_norm;
        //computing the loss gradient inplace(storing in logits)
        if(i == label_id){
          logits[row *N + i] = y_hat - 1.0f;
          loss[row] = -logf(y_hat + 1e-15);
        }
        else{
          logits[row * N + i] = y_hat;
        }
    }
}

// std::vector<torch::Tensor>fused_softcross_forward(torch::Tensor inputs, torch::Tensor weights, torch::Tensor bias, torch::Tensor labels){

//     /*
//     inputs : shape -> (B*S,d)
//     weights : shape -> (d,D)
//     bias : shape -> (D,)
//     */

//     int M = inputs.size(0);
//     int N = weights.size(1);
//     int K = weights.size(0);

//     int threads = 16;

//     //chunking the inputs 
//     int inc_factor = (K + N - 1)/K;
//     int chunk_size =  (M + inc_factor -1) / inc_factor;
//     int num_chunks = (M + chunk_size -1) / chunk_size;

//     float *weight_ptr = weights.data_ptr<float>();
//     float *bias_ptr = bias.data_ptr<float>();

//     auto grad_loss = torch::empty({M,N},inputs.options()); // dl/dz(grad of loss with repsect to logits)
//     float *grad_loss_ptr = grad_loss.data_ptr<float>();  
//     auto loss = torch::empty({M},inputs.options());
//     float *loss_ptr = loss.data_ptr<float>();

//     for(int chunk_id =0 ; chunk_id < num_chunks; chunk_id++){

//         int current_chunk_size = std::min(chunk_size, M - chunk_id * chunk_size);
//         if (current_chunk_size <= 0) break;

//         //chunking the data
//         float *chunked_inputs = inputs.data_ptr<float>() + chunk_id * chunk_size * K;
//         int *chunked_labels_ptr = labels.data_ptr<int>() + chunk_id * chunk_size; 

//         dim3 blockSize(threads,threads);
//         dim3 grid((N + threads -1)/threads, ((current_chunk_size + threads -1)/threads));

//         auto chunked_logits = torch::empty({current_chunk_size,N},inputs.options());
//         float *chunked_logits_ptr = chunked_logits.data_ptr<float>();
//         int smem_size = 2 * threads * threads * sizeof(float);

//         //returns logits
//         matrix_mul_bias<<<grid, blockSize, smem_size>>>(
//           chunked_inputs,weight_ptr, bias_ptr, chunked_logits_ptr,
//           current_chunk_size,N,K,threads
//         );

//         //fused softmax + crossentropy
//         dim3 BlockSize2(256);
//         dim3 grid2(current_chunk_size);

//         auto loss_chunked = torch::empty({current_chunk_size,},inputs.options());

//         float *loss_chunked_ptr = loss_chunked.data_ptr<float>();
//         int smem_size2 = 256 * sizeof(float);

//         fused_crossentropy<<<grid2, BlockSize2, smem_size2>>>(
//           chunked_logits_ptr,chunked_labels_ptr,loss_chunked_ptr,current_chunk_size,N,256
//         );

//         //storing back to original tensor
//         cudaMemcpy(grad_loss_ptr, chunked_logits_ptr, current_chunk_size * N * sizeof(float), cudaMemcpyDeviceToDevice);
//         cudaMemcpy(loss_ptr, loss_chunked_ptr, current_chunk_size * sizeof(float), cudaMemcpyDeviceToDevice);

//         grad_loss_ptr += current_chunk_size  * N;
//         loss_ptr += current_chunk_size ;
//     }
//     return {loss, grad_loss};
// }

/*
-> select some data points (centroids) -> k points
-> calculate the distances of each point with respect to these datapoints
-> put the point to the cluster of min distance
-> after all the points done 
-> calculate the mean of the datapoints to get new centroids
-> again repeat until there is less change in the cluster position 
*/

/*/
GPU strategy 
-> we will have a matrix of shape (N, num of coordiantes) -> (N,2)
-> we will randomly select any k data points (here we use a array of indexes)
-> now each thead will call the shared memory access and store the minimum index
-> we will use an array with N shape with value the cluster index
-> now threads with same cluster index, by mean updates the cluster thing
-> repeats
*/
#define k 3
#define coordinates 2 //number of coordinate axis

__global__ void kmeans (float *input, float *cluster,int *cluster_cd){

    //for storing the cluster coordinates
    __shared__ float smem[k*coordinates];

    int row = threadIdx.x + blockIdx.x * blockDim.x;

    for(int i =0 ; i<k;i++){
      for(int j=0; j<coordinates; j++){
        smem[i*coordinates +j] = cluster_cd[i*coordinates+j];
      }
    }
    __syncthreads();

    int min_idx = INFINITY;
    float min_sum = INFINITY;
    for(int i=0; i<k; i++){
      float sum = 0.0f;
      for(int c =0 ; c<coordinates;c++){
        sum += abs(input[row * coordinates + c] - smem[c + i*coordinates]); //manhattan distance
    }
    if(i<min_idx && min_sum<sum){
      min_idx = i;
    }
  }

  cluster[row] = min_idx;
  __syncthreads();
}

__global__ void partial_sum(float *data,float *cluster_cd, float *cluster,float *partial_sum,float *cluster_count){

    int row = threadIdx.x + blockDim.x *blockIdx.x;
    
    __shared__ float sharedSums[k * coordinates];  
    __shared__ int sharedCounts[k];

    for (int i = threadIdx.x; i < k * coordinates; i += blockDim.x) 
        sharedSums[i] = 0.0f;

    for (int i = threadIdx.x; i < k; i += blockDim.x)
        sharedCounts[i] = 0;

    __syncthreads();

    int cluster_id = cluster[row];
    for (int d = 0; d < coordinates; d++) {
        atomicAdd(&sharedSums[cluster_id * coordinates + d], data[row * coordinates + d]);
    }
    atomicAdd(&sharedCounts[cluster_id], 1);

    __syncthreads();

    for (int i = threadIdx.x; i < k * coordinates; i += blockDim.x) 
        atomicAdd(&partial_sum[i], sharedSums[i]);

    for (int i = threadIdx.x; i < k; i += blockDim.x) 
        atomicAdd(&cluster_count[i], sharedCounts[i]);

    int local_id = threadIdx.x;
    if (local_id < k && cluster_count[local_id] > 0) {
        for (int d = 0; d < coordinates; d++) {
          cluster_cd[local_id * coordinates + d] = partial_sum[local_id * coordinates + d] / cluster_count[local_id];
        }
    }
}

# CUDA Learning Notes – [30-01-2025]

# CUDA Learning - Day 4

## Vector Dot Product
- for naive implementation can use each thread to calculate the partial dot Product
- but during storage of sum to global memory-> race conditions arise -> wrong dot Product
- Solution is to use shared memory to store the dot partial dot product 
- then by partial reduction to get the sum from each block -> then calculate the total sum on cpu
- came to know about the __shfl_down_sync (will try to implemente later on )

## Read chapters of Book(Cuda by example)
- read 5th chapter covering the basics of thread cooperation
- the use of __syncthreads() 
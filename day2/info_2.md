# CUDA Learning Notes – [28-01-2025]

# CUDA Learning - Day 2

## Matrix Addition Using Global Indexing
- In CUDA, global indexing allows each thread to compute a unique element in a matrix.
- The index is calculated using both the block and thread indices, enabling each thread to access a specific row and column in a 2D matrix.


### Global Indexing Formula
To assign the correct element of the matrix to each thread:
- int idx = blockDim.x * blockIdx.x + threadIdx.x; // columns for matrix  
- int idy = blockDim.y * blockIdx.y + threadIdx.y; // rows for matrix
- idx = idx + c * idy (row-major form) c is the number of columns


## Learn about stack vs heap memory(c specific)
- stack memory scopes to local variables (functions)
- heap memory scopes to dynamic memory using malloc and it is persistent 

## explored some strategies to optimize the kernel
- it includes increasing occupancy changing gridsize 
- used cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);


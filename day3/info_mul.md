# CUDA Learning Notes – [29-01-2025]

# CUDA Learning - Day 3

## Matrix Multiplication Using Global Indexing
- Understood the working of 2d paralellism in cuda
- Used 2d grid intialization and 2d thread intialization
- for matrix Multiplication -> each thread calculates the each element of output matrix
- used blockSize of (16,16) -> 256 threads for occupancy
- learned to intialize blocks based on required resources 

## Implementation of ReLU(rectified linear unit)
- each thread calculates the operation of max(0,z)

- started to read another book on cuda(cuda by example)
- read 4 four chapters covering basics



# CUDA Learning Notes – [3-02-2025]

# CUDA Learning - Day 8

## Implemented Softmax
- use shared memory to efficiently store matrix for parallel reduction
- Implementation supports batch size 
- use parallel reduction for calculating the max values and norm values
- each thread calculated the softmax value in the matrix
- managed gridsize and blocksize to work on the matrix

- task is to further optimize it 
- use more efficient parallel reduction than used 
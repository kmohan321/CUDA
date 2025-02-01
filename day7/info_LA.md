# CUDA Learning Notes – [2-02-2025]

# CUDA Learning - Day 7

## Implemented Lightning Attention
- Read the Minimax01 paper to understand the Lightning Attention
- Effecienlty used shared memory to store query , key and values
- Used DRAM for storing the KV values
- Working for if (block size ) (D,B)<1024 (will use tiling for dim direction)
- Further task is to Implement LA2

- task is to further optimize it 
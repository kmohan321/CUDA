# CUDA Learning Notes – [11-02-2025]

## CUDA Learning - Day 16

### Implemented QKV Projection using Tiled Matrix Multiplication

- **Goal**: Efficiently compute **Query (Q), Key (K), and Value (V) projections** from input embeddings using **tiled matrix multiplication** in CUDA.
  
- **Approach**:
  - Used **tiled matrix multiplication** to optimize memory access patterns.
  - Launched **3 separate matrix multiplications** (for Q, K, and V) using the same kernel.
  - Each **batch is handled independently** in the Z-dimension of the grid.
  
### Key Details:

- **Matrix Multiplication (Tiled)**:
  - Each **thread block** computes a tile of the output matrix.
  - Used **shared memory** (`extern __shared__`) to cache tiles of `A` and `B` for reuse.
  - Each thread computes a partial sum, reducing global memory accesses.

- **Grid and Thread Block Configuration**:
  - **Grid Dim:** `(D/blocksize, S/blocksize, B)`  
    - **D** (embedding dimension) is divided across the **X dimension**.
    - **S** (sequence length) is divided across the **Y dimension**.
    - **B** (batch size) is handled in the **Z dimension**.
  - **Block Dim:** `(blocksize, blocksize)` to maximize shared memory efficiency.

- **Shared Memory Utilization**:
  - Used **shared memory** to load input tiles from `A` and `B`.
  - **Avoided bank conflicts** by properly indexing shared memory.
  
- **QKV Projection Steps**:
  1. Compute `Q = X @ W_Q`
  2. Compute `K = X @ W_K`
  3. Compute `V = X @ W_V`
  4. Store the results in separate output matrices.
  
### To Do / Next Steps:
- Implement **Backward Pass** for QKV projections.
- Benchmark **tiled vs. non-tiled matrix multiplication** for performance comparison.
- Profile shared memory usage and optimize further if needed.

### Challenges Faced:
- Managing **batch offsets** correctly to handle **batched matrix multiplication**.
- Handling **edge cases** when the input size isn’t a multiple of blocksize.
- Ensuring **correct thread indexing** when loading tiles from global to shared memory.


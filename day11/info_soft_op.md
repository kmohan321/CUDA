# CUDA Learning Notes – [6-02-2025]

## CUDA Learning - Day 11

### Implemented Optimized Softmax

- **Goal**: Implement an efficient softmax function using CUDA that outperforms PyTorch's implementation.
  
- **Approach**:
  - Implemented a CUDA kernel for computing softmax using **log-sum-exp trick** to improve numerical stability.
  - Optimized memory access by leveraging **shared memory** for max and sum computations.
  - Used **parallel reduction** to efficiently compute `max(x)` and `sum(exp(x - max))`.
  - Implemented a **stride-based approach** to handle **more than 1024 columns** (C > 1024).
  
### Key Details:

- **Thread Indexing**:
  - Used **1D thread blocks** where each thread handles multiple columns using `for (int i = tid; i < C; i += blockDim.x)`.  
  - Avoided **idle threads** by ensuring each thread contributes to the reduction.

- **Parallel Reduction Optimization**:
  - **Step 1**: Each thread computes `max(x)` over its assigned columns.
  - **Step 2**: Uses shared memory reduction to find the global max.
  - **Step 3**: Each thread computes `sum(exp(x - max_x))` and reduces it using shared memory.
  - **Step 4**: Computes final `log(sum_exp) + max_x` to return the result.

- **Shared Memory Usage**:
  - Reduced global memory accesses by storing intermediate results in shared memory.
  - Ensured proper synchronization using `__syncthreads()` during reduction steps.


### To Do / Next Steps:

- **Further Optimization**:
  - **Use Warp-Level Primitives (`__shfl_xor_sync()`)** for even faster reductions.
  - **Reduce Shared Memory Usage** by optimizing data reuse patterns.
  - **Experiment with Different Block Sizes** for further performance improvements.

- **Next Component**:
  - Implement **Linear Layer (Forward + Backpropagation)** as the next step in building a full neural network in CUDA.

### Challenges Faced:
- Ensuring correct reduction across multiple columns while maintaining efficiency.
- Handling **C > 1024** effectively without limiting performance.
- Debugging small numerical differences when comparing to PyTorch.



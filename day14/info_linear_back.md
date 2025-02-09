# CUDA Learning Notes – [09-02-2025]

## CUDA Learning - Day 14

### Implemented Linear Layer Backpropagation

- **Goal**: Implement an efficient backpropagation kernel for a linear layer using CUDA.
  
- **Approach**:
  - Computed gradients for **weights (`dW`)**, **bias (`db`)**, and **input (`dx`)** efficiently using CUDA.
  - Utilized **tiled matrix multiplication** for weight gradient computation.
  - Used **shared memory** for better memory access patterns and improved performance.
  
### Key Details:

- **Gradient Computation**:
  - **Input Gradient (`dx`)**: Computed as `dx = dy * W.T`, leveraging **efficient tiled matrix multiplication**.
  - **Weight Gradient (`dW`)**: Computed using **batched outer product** approach.
  - **Bias Gradient (`db`)**: Reduced along batch dimension using **parallel reduction**.
  
- **Parallelization Strategy**:
  - Each **block processes a tile** of the matrix, reducing global memory accesses.
  - Used **parallel reduction** for bias term computation to avoid redundant computations.

### To Do / Next Steps:
- Implement **further optimizations** using **tensor cores** for faster matrix multiplications.
- Optimize memory access patterns for even better performance.
- Integrate backpropagation with **activation functions** to complete the full layer-wise gradient flow.
  
### Challenges Faced:
- Ensuring correct gradient flow when implementing **weight updates**.
- Optimizing parallel reductions while minimizing synchronization overhead.
- Handling large batch sizes efficiently while keeping memory usage in check.

# CUDA Learning Notes – [17-02-2025]

## CUDA Learning - Day 22

### Implemented Flash Attention in CUDA

- **Goal**: Implement the **Flash Attention** mechanism in CUDA to explore **efficient memory access**, **tiled matrix multiplication**, and **softmax computation**.
- **Concept**:
  - Flash Attention improves upon standard attention by **reducing memory footprint**.
  - Uses **tiling and shared memory** to efficiently compute QK^T and the weighted sum with V.

### **CUDA Kernel Implementation**

- **Approach**:
  - Each thread block processes a **tile of Q, K, and V matrices**.
  - Utilized **shared memory** to reduce redundant memory accesses.
  - Implemented **softmax with numerical stability** using **online normalization**.

- **Key Operations Implemented**:
  1. **Loading Q, K, and V Tiles**: Each tile is stored in **shared memory**.
  2. **Computing Attention Scores (QK^T)**: Using **dot product** across tiles.
  3. **Softmax with Log-Sum-Exp Trick**: For **numerical stability and efficiency**.
  4. **Computing the Final Output (OV Product)**: Using **parallel reduction**.

- **Grid and Block Configuration**:
  - Used **(M, N) grid configuration** where M and N represent **query and key tiles**.
  - Each block processes **a submatrix of QK^T and OV**, utilizing **tiled matrix multiplication**.
  - Implemented **thread synchronization (`__syncthreads()`)** to ensure correct computation.

### **Optimizations & Challenges**

#### **Optimizations Applied**
- **Shared Memory Optimization**:
  - Instead of accessing **global memory repeatedly**, stored **Q, K, and V tiles** in **shared memory**.
  - Ensured **coalesced memory access** while reading/writing from global memory.

- **Numerically Stable Softmax**:
  - Used **max-subtraction trick** for stability.
  - Implemented **online summation** to avoid large exponentials.

- **Efficient Memory Access**:
  - Minimized **bank conflicts** in shared memory.
  - Reduced **global memory transactions** by optimizing **load/store patterns**.
  
#### **Challenges Faced**
- **Incorrect Thread Mapping**: Initially launched threads using `(Br, Bc)` but used **1D indexing inside the kernel**, causing incorrect behavior.
- **Handling Edge Cases**: Needed to ensure proper computation at **tile boundaries**.
- **Ensuring Correct Softmax Computation**: Debugged with **CPU vs. GPU output comparison**.

### **Key Learnings**
- **Shared Memory is Crucial**:
  - Reducing **global memory accesses** significantly improves performance.
  - **Thread synchronization (`__syncthreads()`)** is necessary for correct execution.

- **Efficient Softmax Computation**:
  - Avoiding **large exponentials** and using **log-sum-exp trick** improves stability.

- **CUDA Debugging Techniques**:
  - Used `cudaMemcpy` to **copy GPU results to CPU** for verification.
  - Printed intermediate results to ensure correctness of calculations.

---
Next, I plan to **optimize memory access patterns** further and **benchmark performance** against standard attention. 🚀
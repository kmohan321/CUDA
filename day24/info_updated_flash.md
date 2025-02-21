# CUDA Learning Notes – [19-02-2025]

## CUDA Learning - Day 24

### Implemented Flash Attention in CUDA

- **Goal**: Implement an optimized **Flash Attention kernel** in CUDA that efficiently computes attention scores while reducing memory overhead.
- **Concept**:
  - Flash Attention reduces memory usage by computing softmax and weighted sums in a **block-wise streaming fashion**.
  - Utilizes **tiling**, **shared memory**, and **parallel reduction** for efficient computation.
  - Computes **attention scores** using dot product scaling and normalization within shared memory.

### **CUDA Kernel Implementation**

- **Approach**:
  - Each thread block processes **tile-wise updates** for queries (Q), keys (K), and values (V).
  - Uses **shared memory** to store intermediate results like scaled dot products and softmax values.
  - Implements **parallel reduction** to compute row-wise max and sum for stable softmax computation.
  - Updates the output matrix (O) efficiently by integrating previous max and sum values.

- **Key Operations Implemented**:
  1. **Tile Loading**:
     - Threads load **Q, K, and V** tiles from global memory into **shared memory**.
     - Ensures **coalesced access** to reduce memory latency.
  2. **Attention Score Computation**:
     - Compute **S = QK^T * scale** in shared memory.
     - Uses **parallel reduction** to compute **row-wise max and sum**.
  3. **Softmax Calculation**:
     - Normalizes **attention scores using row-wise max and sum** for numerical stability.
  4. **Weighted Sum Calculation**:
     - Computes **final output O = Softmax(S) * V**.
     - Uses **thread-wise accumulation** to update `O` efficiently.
  5. **Stateful Update of Running Max and Sum**:
     - Tracks **previous max and sum values** to enable block-wise streaming.

- **Grid and Block Configuration**:
  - Uses a **(batch, head) grid configuration**.
  - Each **thread block processes a (B_r, B_c) tile**, where:
    - `B_r`: Number of rows per tile (query chunk size)
    - `B_c`: Number of columns per tile (key chunk size)
  - Uses **T_r x T_c thread layout** to handle different matrix dimensions efficiently.

### **Optimizations & Challenges**

#### **Optimizations Applied**
- **Shared Memory Optimization**:
  - Stored **Q, K, V, and S** in shared memory to reduce redundant global memory accesses.
  - Applied **parallel reduction** for efficient **row-wise max and sum** computation.
- **Efficient Memory Access**:
  - Used **coalesced access** while loading tiles from global to shared memory.
  - Reduced **global memory transactions** by optimizing **load/store patterns**.
- **Parallel Computation for Softmax**:
  - Performed **log-sum-exp trick** using row-wise max.
  - Normalized in **shared memory** before updating global memory.

#### **Challenges Faced**
- **Numerical Stability in Softmax**: Needed to carefully handle **row-wise max subtraction** to avoid overflow issues.
- **Handling Tile Boundaries**: Ensured correct computation when the number of elements was not a multiple of `B_r` or `B_c`.
- **Synchronization Issues**: Debugged **race conditions in reduction** using `__syncthreads()`.
- **Thread Divergence in Reduction**: Optimized to minimize **warp divergence** during summation.

### **Key Learnings**
- **Block-wise Processing Reduces Memory Footprint**:
  - Streaming attention computation **avoids large intermediate tensors**.
  - Efficiently integrates into larger Transformer architectures.
- **Parallel Reduction is Crucial for Softmax Stability**:
  - **Using row-wise max** prevents numerical instability.
  - Ensures correctness when handling small probability values.
- **CUDA Debugging Techniques**:
  - Compared against PyTorch’s attention outputs for verification.
  - Used **cudaMemcpy** to check intermediate shared memory values.
  - Profiling with **Nsight Compute** helped optimize memory access patterns.

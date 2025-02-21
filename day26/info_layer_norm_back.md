# CUDA Learning Notes – [21-02-2025]

## CUDA Learning - Day 26

### Implemented Layer Normalization Backward in CUDA

- **Goal**: Implement the **backward pass of Layer Normalization** in CUDA to efficiently compute gradients for **input, scale (gamma), and shift (beta)** parameters.
- **Concept**:
  - LayerNorm requires computing gradients for **input (dx)**, **gamma (dgamma)**, and **beta (dbeta)**.
  - Utilizes **shared memory reduction** to accumulate gradient sums efficiently.

### **CUDA Kernel Implementation**

- **Approach**:
  - Each thread block processes a **row of the input matrix**.
  - Utilized **shared memory** to perform **parallel reduction** for summing gradients.
  - Implemented **numerically stable gradient computation** using **precomputed mean and variance**.

- **Key Operations Implemented**:
  1. **Compute `dbeta` and `dgamma`**: Per-feature accumulation using **parallel reduction**.
  2. **Compute `dx`**: Uses shared memory reduction to normalize gradients.
  3. **Efficiently apply `gamma` scaling** while computing the final `dx`.

- **Grid and Block Configuration**:
  - Used **(N) grid configuration**, where each block processes a **single row**.
  - Threads iterate over **features (D)** in a **tiled manner** to ensure full coverage.
  - Implemented **thread synchronization (`__syncthreads()`)** to avoid race conditions.

### **Optimizations & Challenges**

#### **Optimizations Applied**
- **Shared Memory Optimization**:
  - Used shared memory to **reduce redundant global memory accesses**.
  - Applied **parallel reduction** for efficient `sum(dhatx)` and `sum(dhatx * xmu)` computation.

- **Efficient Memory Access**:
  - Ensured **coalesced memory access** while reading input and gradients.
  - Reduced **global memory transactions** by optimizing **load/store patterns**.

#### **Challenges Faced**
- **Incorrect Shared Memory Reduction**: Initial implementation did not synchronize correctly, causing incorrect sums.
- **Numerical Stability in Normalization**: Needed to carefully apply **sigma inverse scaling**.
- **Ensuring Correct Thread-Wise Updates**: Debugged using **CPU verification** against PyTorch’s LayerNorm backward.

### **Key Learnings**
- **Parallel Reduction is Essential**:
  - Efficient summation of **dhatx** and **dhatx * xmu** avoids per-thread redundant computations.
  - **Thread synchronization (`__syncthreads()`)** ensures correctness.

- **Memory Access Patterns Matter**:
  - Avoiding **strided memory access** improves kernel efficiency.
  - Using **shared memory for intermediate sums** reduces global memory bottlenecks.

- **CUDA Debugging Techniques**:
  - Compared **CPU vs. GPU outputs** to validate gradient computations.
  - Used **cudaMemcpy** to transfer intermediate values for Debugging


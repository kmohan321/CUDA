# CUDA Learning Notes – [Date]

## CUDA Learning - Day [X]

### Implemented Basic Attention Mechanism in CUDA

- **Goal**: Implement the **basic attention mechanism** in CUDA to understand **matrix multiplications**, **softmax computation**, and **parallelization techniques**.
- **Concept**:
  - Computes **dot-product attention** between query (Q), key (K), and value (V) matrices.
  - Uses **softmax normalization** to weigh the values dynamically.

### **CUDA Kernel Implementation**

- **Approach**:
  - Each thread computes a single **(row, column)** element in the attention matrix.
  - Utilized **shared memory** to store intermediate **attention scores**.
  - Performed **matrix-vector multiplications** efficiently using CUDA threads.

- **Computation Steps**:
  1. Compute **dot-product attention scores**: \( S_{ij} = Q_i \cdot K_j^T \)
  2. Apply **softmax normalization** to obtain weights.
  3. Compute final output **O = softmax(S) \cdot V**.

- **Grid and Block Configuration**:
  - Each thread computes a single **(row, col) value** in **(sequence_length, feature_dim)**.
  - Used **(blockDim.x, blockDim.y) thread blocks** for parallelization.

### **Optimizations & Challenges**

#### **Optimizations Applied**
- **Shared Memory Optimization**:
  - Used **extern shared memory** for storing **attention scores**, reducing global memory accesses.
  - Ensured **coalesced memory access** in softmax computation to improve efficiency.

- **Efficient Softmax Computation**:
  - Used **max-subtraction trick** to improve numerical stability during exponentiation.
  - Performed **normalization efficiently** in parallel.

#### **Challenges Faced**
- **Race conditions** due to **shared memory updates**.
- Needed to ensure **correct indexing** in matrix computations.
- Debugging **numerical stability issues** in softmax calculation.


### **Key Learnings**
- **Parallel Matrix Multiplication**:
  - Each thread computes a single element in a large dot-product operation.
  - CUDA enables **highly parallel execution**, improving performance over CPU methods.

- **Softmax Optimization Techniques**:
  - **Using shared memory** reduces global memory latency.
  - **Using max subtraction** helps prevent floating-point underflow.

- **CUDA Debugging Techniques**:
  - Verified results by comparing CUDA outputs with **CPU-based implementations**.
  - Used **printf debugging** to check memory access patterns.

---
Next, I plan to **optimize memory access further** and compare performance against cuBLAS. 🚀
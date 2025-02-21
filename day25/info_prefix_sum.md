# CUDA Learning Notes – [20-02-2025]

## CUDA Learning - Day 25

### Implemented Prefix Sum (Scan) in CUDA

- **Goal**: Implement the **prefix sum (exclusive scan)** in CUDA using **shared memory** and **block-level reduction**.
- **Concept**:
  - Each **CUDA block** processes a segment of the input array.
  - Uses **shared memory** for fast intra-block scan computation.
  - A **second kernel** handles block-level sum propagation for final results.

### **CUDA Kernel Implementation**

#### **Step 1: Intra-Block Prefix Sum (Shared Memory)**
- Each block loads a segment into **shared memory**.
- Performs **in-place parallel prefix sum** using **stride-based updates**.
- The last thread stores the **block sum** for later correction.

#### **Step 2: Block-Level Sum Propagation**
- A second kernel adds the **prefix sum of previous blocks** to each element.
- Ensures correctness across **multiple blocks**.

### **Kernel Code Overview**

#### **1. Prefix Sum Kernel (`prefix_sum`)**
- **Loads input into shared memory** for fast computation.
- Uses a **parallel scan** approach with stride-based updates.
- Stores **block sums** for later adjustment.

#### **2. Block-Level Sum Correction Kernel (`block_reduction`)**
- Propagates **block sums** across the entire output.
- Ensures correctness across multiple CUDA blocks.

### **Grid and Block Configuration**
- **Thread Block Size (`smem_size = 32`)**
- **Handles `N = 100` elements** using **multiple blocks**.
- Uses **coalesced memory access** for efficient global memory reads/writes.

### **Optimizations & Challenges**

#### **Optimizations Applied**
- **Shared Memory Optimization**:
  - Reduced global memory transactions by leveraging **shared memory** for intra-block computation.
- **Parallel Reduction for Efficient Computation**:
  - Implemented an optimized **stride-based sum** approach.
- **Minimized Divergence**:
  - Used **if-else guards** to prevent out-of-bounds memory access.

#### **Challenges Faced**
- **Handling Multiple Blocks**:
  - Ensured correct sum propagation across different blocks.
- **Shared Memory Updates**:
  - Needed correct **synchronization (`__syncthreads()`)** to avoid race conditions.
- **Edge Cases**:
  - Addressed cases where `N` is not a multiple of `smem_size`.

### **Key Learnings**
- **Shared Memory Boosts Performance**:
  - Reduced global memory accesses significantly.
- **Block-Level Sum Propagation is Necessary**:
  - Required a second kernel for correctness.
- **Correct Thread Synchronization (`__syncthreads()`) is Critical**:
  - Prevented race conditions in shared memory updates.

### **Next Steps**
- Implement a **more efficient, work-efficient scan algorithm** (e.g., **Blelloch scan**).
- Compare **performance against thrust::inclusive_scan**.
- Extend to **multi-GPU parallel prefix sum**.

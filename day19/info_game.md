# CUDA Learning Notes – [14-02-2025]

## CUDA Learning - Day 19

### Implemented Conway's Game of Life in CUDA

- **Goal**: Implement the **Game of Life** simulation in CUDA to explore **shared memory usage**, **parallel computation**, and **neighbor-based rules**.
- **Concept**:
  - Each cell in a grid follows simple birth and survival rules.
  - Uses **neighbor counting** to determine the next state of each cell.

### **CUDA Kernel Implementation**

- **Approach**:
  - Each thread processes **one cell** and computes its **next state**.
  - Utilized **shared memory** to store local block data for efficient neighbor access.
  - Handled **boundary conditions** to avoid out-of-bounds memory accesses.

- **Rules Implemented**:
  1. **Underpopulation**: A live cell with fewer than **2 live neighbors** dies.
  2. **Survival**: A live cell with **2 or 3 live neighbors** stays alive.
  3. **Overpopulation**: A live cell with more than **3 live neighbors** dies.
  4. **Reproduction**: A dead cell with exactly **3 live neighbors** becomes alive.

- **Grid and Block Configuration**:
  - Used **(M, N) grid configuration** to process large grids.
  - Each block processes a **sub-grid** and utilizes **shared memory** for fast access.
  - Implemented **thread synchronization (`__syncthreads()`)** for correct results.

### **Optimizations & Challenges**

#### **Optimizations Applied**
- **Shared Memory Optimization**:
  - Instead of accessing **global memory for each neighbor**, used **shared memory** to reduce memory latency.
  - Avoided bank conflicts by **aligning shared memory accesses**.
  
- **Efficient Memory Access**:
  - Ensured **coalesced memory access** for global memory reads/writes.
  - Loaded data in **contiguous memory layout** to maximize performance.
  
#### **Challenges Faced**
- Initially, **included the cell itself while counting neighbors**, which caused incorrect results.
- Needed to properly **handle edge cases** at block boundaries.
- Debugging **GPU results vs. CPU results** to ensure correctness.


### **Key Learnings**
- **Importance of Shared Memory**:
  - Using shared memory **greatly reduces global memory accesses**.
  - Synchronization (`__syncthreads()`) is essential for correctness.
  
- **Correct Handling of Neighbors**:
  - Ensuring correct **neighbor summation** is crucial for expected game behavior.
  
- **CUDA Debugging Techniques**:
  - Used `cudaMemcpy` to **copy GPU results to CPU** for comparison.
  - Printed intermediate results to verify correctness.

---
Next, I plan to scale up the simulation and visualize the output using **OpenGL or Matplotlib** for better analysis. 🚀


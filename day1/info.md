# CUDA Learning Notes – [Date]

## 1. Understanding Memory Allocation
- **Host vs Device Memory:**  
  - Arrays like `int vec1[M]` are allocated on the **host (CPU)** memory.  
  - GPU memory is allocated using `cudaMalloc()` and accessed via pointers.  
  - Data transfer between host and device is done via `cudaMemcpy()`.

## 2. Kernel Execution
- **Launching Kernels:**  
  - Kernels are executed on the GPU using a grid-block-thread hierarchy.  
  - Example:  
    ```cpp
    vec_add<<<blocks, threads_per_block>>>();
    ```
  - Choosing the right block and thread count is critical for performance.

## 3. Optimizing Execution
- **Grid and Block Sizing:**  
  - The number of blocks should be calculated to cover all elements:  
    ```cpp
    int blocks = (M + m - 1) / m;
    ```  
  - The choice of `m` (threads per block) affects GPU occupancy.

- **Profiling Performance:**  
  - CUDA kernels can be profiled to analyze execution time and memory usage.  
  - Tools like `nvprof` or `Nsight Compute` can be used to check kernel performance.  
  - Profiling helps identify bottlenecks such as memory bandwidth or compute limitations.

## 4. Debugging Performance Issues
- **Possible Reasons for No Speedup:**  
  - Small problem sizes may not fully utilize GPU resources.  
  - Kernel launch overhead may overshadow computation for simple operations.  
  - Inefficient memory access patterns can cause high latency.  
  - Low occupancy if too few threads are running.

## 5. Improving Performance
- **Strategies Explored:**  
  - Increasing problem size (`M`) to leverage GPU parallelism.  
  - Experimenting with different block sizes to maximize occupancy.  
  - Ensuring memory accesses are coalesced for better performance.  
  - Using shared memory for frequently accessed data to reduce global memory latency.

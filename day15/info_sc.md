# CUDA Learning Notes – [10-02-2025]

## CUDA Learning - Day 15

### Implemented Softmax with Cross-Entropy Forward Pass

- **Goal**: Implement an efficient Softmax with Cross-Entropy forward pass using CUDA.
  
- **Approach**:
  - Computed **Softmax probabilities** in a numerically stable way using **online softmax**.
  - Used **parallel reduction** to compute **max-logits** and **normalization factor**.
  - Calculated **cross-entropy loss** using the true class probabilities.
  
### Key Details:

- **Softmax Computation**:
  - **Numerically stable max subtraction**: First, compute `max(logits)` per row to avoid overflow issues.
  - **Exponentiation and normalization**: Apply `exp(logit - max_logit)` and compute the sum for normalization.
  - **Parallel reduction**: Reduce across threads to compute the final normalization factor efficiently.
  
- **Cross-Entropy Loss Calculation**:
  - Extract the **logit corresponding to the true label**.
  - Compute `loss[row] = -log(softmax[true_label])`.
  
- **Parallelization Strategy**:
  - Each **block processes one row** of the logits matrix.
  - Used **shared memory** for intermediate reductions to improve efficiency.
  - **Minimized global memory accesses** by storing intermediate results in shared memory.
  
### To Do / Next Steps:
- Optimize **shared memory usage** to further reduce memory overhead.
- Explore **alternative reduction strategies** to improve performance.
- Integrate the softmax forward pass with **full model training**.
  
### Challenges Faced:
- Ensuring numerical stability when computing **softmax probabilities**.
- Handling **thread divergence** while performing **reduction operations**.
- Optimizing memory accesses to prevent **global memory bottlenecks**.


# CUDA Learning Notes – [13-02-2025]

## CUDA Learning - Day 18

### Implemented GELU Activation Function in CUDA

- **Goal**: Implement the **GELU activation function** in CUDA and understand its **forward and backward pass**.
- **Formula Used**:
  
  \[
  GELU(x) = 0.5 x \times \left(1 + \tanh \left(\sqrt{\frac{2}{\pi}} \left( x + 0.044715 x^3 \right) \right) \right)
  \]

### **Forward Pass Implementation**
- **Approach**:
  - Each thread computes GELU activation for an **element-wise input**.
  - Used **tanh approximation** to avoid expensive exponentiation.
  - Ensured correct memory access patterns to avoid divergence.
- **Grid and Block Configuration**:
  - Used **(M, N) grid configuration** for handling 2D matrices.
  - Each thread computes one element of the output.
  
### **Backward Pass Implementation**
- **Derivative of GELU**:
  
  \[
  \frac{d}{dx} GELU(x) = 0.5 \left( 1 + \tanh \left( \sqrt{\frac{2}{\pi}} \left( x + 0.044715 x^3 \right) \right) \right) +
  \frac{x \cdot (1 - \tanh^2(\dots)) \cdot \sqrt{\frac{2}{\pi}} (1 + 3 \times 0.044715 x^2)}{2}
  \]

- **Gradient Computation**:
  - `grad_output` is the gradient **received from the next layer**.
  - We compute `grad_input` by multiplying `grad_output` with **dGELU/dx**.
  - Used **element-wise operations** to compute derivatives efficiently.
- **Memory and Performance Optimization**:
  - Used **coalesced memory access** for better performance.
  - Ensured **batch-wise computation** using grid offsets.

### **Key Observations**
- **Handling `grad_output`**:
  - It has the **same shape** as the input and output tensors.
  - Needed for **computing chain rule in backpropagation**.
- **Performance Considerations**:
  - Implemented **efficient indexing** to avoid unnecessary memory accesses.

### **Challenges Faced**
- Understanding the **role of `grad_output`** in backpropagation.
- Correctly implementing **tanh-based derivative calculations**.
- Ensuring **numerical stability** for very small or large input values.



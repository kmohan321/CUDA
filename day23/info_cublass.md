# CUDA Learning Notes – [18-02-2025]

## CUDA Learning - Day 23

### **Explored cuBLAS Library for Vector Operations**

#### **1. Implemented SAXPY using cuBLAS**
- Used `cublasSaxpy` to compute: 
  
  \[ y = a \times x + y \]  
  
- Allocated memory on GPU using `cudaMalloc`
- Copied data between CPU and GPU using `cublasSetVector` and `cublasGetVector`
- Verified results by printing updated `y` values

#### **2. Implemented Dot Product using cuBLAS**
- Used `cublasSdot` to compute the dot product:
  
  \[ result = x \cdot y \]
  
- Initialized vectors `x` and `y` with all ones
- Performed dot product operation and retrieved results to CPU
- Printed the computed dot product value

### **Key Takeaways**
- **cuBLAS provides optimized BLAS functions for vector and matrix operations.**
- **Memory management is crucial**: Allocating and deallocating device memory properly avoids memory leaks.
- **Efficient data transfer**: Using `cublasSetVector` and `cublasGetVector` ensures efficient copying between CPU and GPU.
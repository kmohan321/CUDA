import torch
import triton
import triton.language as tl 

@triton.jit
def online_softmax(
  X_ptr : torch.Tensor,
  Y_ptr : torch.Tensor,
  N : int,
  stride : int,
  blocksize : tl.constexpr
):
  
  row_id = tl.program_id(0);
  
  # starting row
  X_ptr += row_id * stride;
  Y_ptr += row_id * stride; 
  
  local_max = -float("inf")
  local_norm = 0.0
  for i in range(0,N,blocksize):
    n_off = tl.arange(0,blocksize) + i 
    mask = n_off < N
    curr_value = tl.load(X_ptr + n_off, mask=mask,other=-float("inf"))
    new_local_max = tl.max(curr_value)
    local_norm = local_norm * tl.exp(local_max - new_local_max) + tl.sum(tl.exp(curr_value-new_local_max),axis=0)
    local_max = new_local_max #storing the current local max
  

  for i in range(0,N,blocksize):
    n_off = tl.arange(0,blocksize) + i
    mask = n_off < N 
    curr_value = tl.load(X_ptr + n_off, mask=mask)
    softmax_values = tl.exp(curr_value-local_max) / local_norm
    tl.store(Y_ptr + n_off,softmax_values,mask=mask)
    
def softmax_triton(X):
    """Wrapper to call Triton kernel for row-wise softmax"""
    B, N = X.shape
    Y = torch.empty_like(X)

    grid = (B,)  # One block per row
    stride = X.stride(0)

    online_softmax[grid](X, Y, N, stride, blocksize=128)
    return Y

# Example Usage
X = torch.randn(2, 4, device="cuda", dtype=torch.float32)
Y = softmax_triton(X)
y_torch = torch.softmax(X,axis=1)
print(torch.allclose(Y,y_torch,atol=1e-5))
print((y_torch-Y).abs().max())
print("Input:\n", X)
print("Softmax Output:\n", Y)
print(torch.sum(Y,axis=1))

    
  
  
  
  
  
  
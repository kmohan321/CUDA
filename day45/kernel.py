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
  
  X_ptr += row_id * stride;
  Y_ptr += row_id * stride; # starting row
  
  block_max = tl.full([blocksize],-float("inf"),dtype=tl.float32)
  block_norm = tl.zeros([blocksize],dtype=tl.float32)

  for i in range(0,N,blocksize):
    n_off = tl.arange(0,blocksize) + i 
    mask = n_off < N
    curr_value = tl.load(X_ptr + n_off, mask=mask,other=-float("inf"))
    block_max = tl.maximum(block_max,curr_value)
    block_norm = block_norm * tl.exp(block_max-curr_value)
    block_norm += tl.exp(curr_value-block_max)
    
  global_max = tl.max(block_max,axis=0)
  block_norm = block_norm * tl.exp(block_max-global_max)
  global_norm = tl.sum(block_norm,axis=0)
  
  for i in range(0,N,blocksize):
    n_off = tl.arange(0,blocksize) + i
    mask = n_off < N 
    curr_value = tl.load(X_ptr + n_off, mask=mask)
    softmax_value = tl.exp(curr_value-global_max)/global_norm
    tl.store(Y_ptr + n_off,softmax_value,mask=mask)
    
def softmax_triton(X):
    """Wrapper to call Triton kernel for row-wise softmax"""
    B, N = X.shape
    Y = torch.empty_like(X)

    grid = (B,)  # One block per row
    stride = X.stride(0)

    online_softmax[grid](X, Y, N, stride, blocksize=1024)
    return Y

# Example Usage
X = torch.randn(4, 10, device="cuda", dtype=torch.float32)
Y = softmax_triton(X)

print("Input:\n", X)
print("Softmax Output:\n", Y)

    
  
  
  
  
  
  
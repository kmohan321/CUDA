import torch
import triton
import triton.language as tl
import time

@triton.jit
def layer_norm_optimized(
    X, Y,
    gamma, beta,
    stride, N,
    BLOCK_SIZE: tl.constexpr,
    eps
):
    row_idx = tl.program_id(0) #we are working with one row for each block
    row_offset = row_idx * stride
    
    local_mean = tl.zeros([BLOCK_SIZE],dtype=tl.float32)
    #striding over the feature dimension
    for i in range(0,N,BLOCK_SIZE):
      #simply calling all the indexes
      n_off = tl.arange(0,BLOCK_SIZE) + i
      x = tl.load(X + row_offset + n_off, mask = n_off<N, other=0)
      local_mean += x 
    global_mean = tl.sum(local_mean,axis=0)/N
    
    local_variance = tl.zeros([BLOCK_SIZE],dtype=tl.float32)
    #striding over the feature dimension
    for i in range(0,N,BLOCK_SIZE):
      #simply calling all the indexes
      n_off = tl.arange(0,BLOCK_SIZE) + i
      x = tl.load(X + row_offset + n_off, mask = n_off<N, other=0)
      local_variance += (x - global_mean) *(x-global_mean)
    global_variance = tl.sum(local_variance,axis=0)/N

    rstd = 1 / tl.sqrt(global_variance + eps)
    
    for i in range(0,N,BLOCK_SIZE):
      n_off = tl.arange(0,BLOCK_SIZE) + i
      gamma_ptrs = gamma + n_off
      beta_ptrs = beta + n_off
      g = tl.load(gamma_ptrs, mask=n_off<N, other=0.0)
      b = tl.load(beta_ptrs, mask=n_off<N, other=0.0)
      x = tl.load(X + row_offset + n_off, mask = n_off<N, other=0)
      x_norm = (x - global_mean) * rstd
      y = g * x_norm + b
      y_ptrs = Y + row_offset + n_off
      tl.store(y_ptrs, y, mask=n_off<N)

def layer_norm(X,Y,gamma,beta,Blocksize,eps):
  
    M,N = X.shape
    grid = (M,1)
    layer_norm_optimized[grid](X,Y,gamma,beta,X.stride(0),N,Blocksize,eps)
    
    return Y
  
def benchmark_layer_norm(layer_norm_fn, M=512, N=2048, blocksize=512, eps=1e-5):
    X = torch.randn(size=(M, N), device='cuda')
    Y = torch.empty_like(X)  
    gamma = torch.ones(size=(N,), device='cuda')
    beta = torch.zeros(size=(N,), device='cuda')

    torch.cuda.synchronize()
    start_time = time.time()
    y_custom = layer_norm_fn(X, Y, gamma, beta, blocksize, eps)
    torch.cuda.synchronize()
    custom_time = time.time() - start_time

    ln = torch.nn.LayerNorm(N, device='cuda', eps=eps)
    torch.cuda.synchronize()
    start_time = time.time()
    y_torch = ln(X)
    torch.cuda.synchronize()
    torch_time = time.time() - start_time

    max_diff = torch.max(torch.abs(y_custom - y_torch)).item()
    correctness = torch.allclose(y_custom, y_torch, atol=1e-5)

    print(f"Custom LayerNorm Time: {custom_time:.6f}s")
    print(f"PyTorch LayerNorm Time: {torch_time:.6f}s")
    print(f"Max Difference: {max_diff}")
    print(f"Correctness: {'✓' if correctness else '✗'}")

    return custom_time, torch_time, correctness
if __name__ == '__main__':
  benchmark_layer_norm(layer_norm)
  
    



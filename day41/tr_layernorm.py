import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_optimized(
    X, Y,
    gamma, beta,
    stride, N,
    BLOCK_SIZE: tl.constexpr,
    eps
):
    row_idx = tl.program_id(0)
    row_offset = row_idx * stride
    
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = X + row_offset + offsets
    
    mask = offsets < N
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    
    mean = tl.sum(x, axis=0) / N
    
    x_centered = x - mean
    x_centered = tl.where(mask, x_centered, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps) 
    x_norm = x_centered * rstd
    
    gamma_ptrs = gamma + offsets
    beta_ptrs = beta + offsets
    g = tl.load(gamma_ptrs, mask=mask, other=0.0)
    b = tl.load(beta_ptrs, mask=mask, other=0.0)
    
    y = g * x_norm + b
    y_ptrs = Y + row_offset + offsets
    tl.store(y_ptrs, y, mask=mask)
    



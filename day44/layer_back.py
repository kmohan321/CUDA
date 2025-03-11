import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_backward_kernel(
    DX, DY, X, MU, VAR, GAMMA, 
    D, EPS, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    offset = row * D + tl.arange(0, BLOCK_SIZE)
    mask = offset < X.shape[0] * D
    
    x = tl.load(X + offset, mask=mask, other=0)
    dy = tl.load(DY + offset, mask=mask, other=0)
    gamma = tl.load(GAMMA + tl.arange(0, BLOCK_SIZE), mask=mask, other=1)
    mu = tl.load(MU + row)
    var = tl.load(VAR + row)
    
    std_inv = 1 / tl.sqrt(var + EPS)
    x_hat = (x - mu) * std_inv
    d_hat = dy * gamma
    
    # Compute mean(d_hat) and mean(d_hat * x_hat)
    mean_d_hat = tl.sum(d_hat, axis=0) / D
    mean_d_hat_x_hat = tl.sum(d_hat * x_hat, axis=0) / D
    
    # Compute dX
    dx = (d_hat - mean_d_hat - x_hat * mean_d_hat_x_hat) * std_inv
    
    # Store the result
    tl.store(DX + offset, dx, mask=mask)

def layer_norm_backward(dx, dy, x, mu, var, gamma, eps=1e-5):
    B, D = x.shape
    dx = torch.empty_like(x)
    
    layer_norm_backward_kernel[(B,)](
        dx, dy, x, mu, var, gamma, D, eps, 
        BLOCK_SIZE=triton.next_power_of_2(D)
    )
    
    return dx

import triton
import triton.language as tl
import torch

@triton.jit
def rmsnorm_fwd_kernel(X, Gamma, Y, EPSILON, BLOCK_SIZE: tl.constexpr, N_FEATURES: tl.constexpr):

    row = tl.program_id(0)
    
    x_ptrs = X + row * N_FEATURES + tl.arange(0, BLOCK_SIZE)
    gamma_ptrs = Gamma + tl.arange(0, BLOCK_SIZE)
    y_ptrs = Y + row * N_FEATURES + tl.arange(0, BLOCK_SIZE)

    x = tl.load(x_ptrs, mask=tl.arange(0, BLOCK_SIZE) < N_FEATURES, other=0.0)
    
    rms = tl.sqrt(tl.sum(x * x) / N_FEATURES + EPSILON)

    gamma = tl.load(gamma_ptrs, mask=tl.arange(0, BLOCK_SIZE) < N_FEATURES, other=1.0)
    y = x / rms * gamma

    tl.store(y_ptrs, y, mask=tl.arange(0, BLOCK_SIZE) < N_FEATURES)

def rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps=1e-5):
    B, D = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(D)

    rmsnorm_fwd_kernel[(B,)]( 
        x, gamma, y,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        N_FEATURES=D,
    )
    return y

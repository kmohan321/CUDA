import triton
import triton.language as tl

@triton.jit
def ddim_triton_kernel(x_ptr, noise_ptr, alpha_ptr, sigma_ptr, t_ptr, N, BLOCK_SIZE: tl.constexpr):
    """DDIM sampler kernel"""
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE

    # Load data
    x_t = tl.load(x_ptr + offset)
    noise_pred = tl.load(noise_ptr + offset)
    alpha_t = tl.load(alpha_ptr + offset)
    sigma_t = tl.load(sigma_ptr + offset)

    x_new = alpha_t * x_t + sigma_t * noise_pred

    # Store result
    tl.store(x_ptr + offset, x_new)


def ddim_triton_sampler(model, x_T, timesteps, alpha_t, sigma_t):
    N = x_T.shape[0]  
    BLOCK_SIZE = 128  

    for t in reversed(range(timesteps)):
        noise_pred = model(x_T, t) 
        ddim_triton_kernel[(N,)](x_T, noise_pred, alpha_t, sigma_t, t, N, BLOCK_SIZE)
    
    return x_T
  


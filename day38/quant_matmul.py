import triton
import triton.language as tl
import torch

@triton.jit
def matmul_kernel(
    A, B, C, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    """ Compute C = A @ B """
    pid_m = tl.program_id(0) #simply blockids 
    pid_n = tl.program_id(1) 
    
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M) #rowoftheout
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) #coloftheout
    off_k = tl.arange(0, BLOCK_SIZE_K) 
    
    a_ptrs = A + off_m[:, None] * stride_am + off_k[None, :] * stride_ak
    b_ptrs = B + off_k[:, None] * stride_bk + off_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K): #tiles
        a = tl.load(a_ptrs, mask=off_m[:, None] < M, other=0)
        b = tl.load(b_ptrs, mask=off_n[None, :] < N, other=0)
        acc += tl.dot(a, b)
        
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    c_ptrs = C + off_m[:, None] * stride_cm + off_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(off_m[:, None] < M) & (off_n[None, :] < N))


M, N, K = 128, 128, 128
BLOCK_SIZE = 16

torch.manual_seed(0)
A = torch.randn((M, K), dtype=torch.float16, device='cuda')
B = torch.randn((K, N), dtype=torch.float16, device='cuda')
C = torch.empty((M, N), dtype=torch.float32, device='cuda')

grid = (M // BLOCK_SIZE, N // BLOCK_SIZE)
matmul_kernel[grid] (
    A, B, C, M, N, K, A.stride(0), A.stride(1), B.stride(0), B.stride(1), C.stride(0), C.stride(1),
    BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE
)

print("Triton Matmul Output:", C)
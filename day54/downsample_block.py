import torch
import triton
import triton.language as tl

@triton.jit
def optimized_conv2d_kernel(
    input_ptr, weight_ptr, output_ptr, 
    H, W, KH, KW, C_in, C_out, stride, 
    BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr
):
    row = tl.program_id(0)  # Output height index
    col = tl.program_id(1)  # Output width index
    co = tl.program_id(2)   # Output channel index (C_out)

    # Define shared memory for input patch
    input_shared = tl.zeros([BLOCK_H + KH - 1, BLOCK_W + KW - 1, C_in], dtype=tl.float32)

    # Load input into shared memory
    for ci in range(C_in):
        for kh in range(KH):
            for kw in range(KW):
                h_idx = row * stride + kh
                w_idx = col * stride + kw
                if h_idx < H and w_idx < W:
                    inp_idx = ci * H * W + h_idx * W + w_idx
                    input_shared[kh, kw, ci] = tl.load(input_ptr + inp_idx)

    # Synchronize shared memory
    tl.sync()

    # Compute convolution
    acc = 0.0
    for ci in range(C_in):
        for kh in range(KH):
            for kw in range(KW):
                wgt_idx = co * C_in * KH * KW + ci * KH * KW + kh * KW + kw
                acc += input_shared[kh, kw, ci] * tl.load(weight_ptr + wgt_idx)

    # Store output
    out_idx = co * (H // stride) * (W // stride) + row * (W // stride) + col
    tl.store(output_ptr + out_idx, acc)

def optimized_conv2d_triton(input_tensor, weight_tensor, stride=1):
    B, C_in, H, W = input_tensor.shape
    C_out, _, KH, KW = weight_tensor.shape

    H_out, W_out = H // stride, W // stride
    output = torch.empty((B, C_out, H_out, W_out), device="cuda")

    grid = (H_out, W_out, C_out)
    
    optimized_conv2d_kernel[grid](
        input_tensor, weight_tensor, output, 
        H, W, KH, KW, C_in, C_out, stride, 
        BLOCK_H=8, BLOCK_W=8, BLOCK_C=16
    )
    return output

# Testing
input_tensor = torch.randn(1, 3, 32, 32, device="cuda")
weight_tensor = torch.randn(16, 3, 3, 3, device="cuda")
output = optimized_conv2d_triton(input_tensor, weight_tensor, stride=2)
print(output.shape)  # Expected: (1, 16, 16, 16)

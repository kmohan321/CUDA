import torch
import triton
import triton.language as tl 

@triton.jit
def dyt_fwd_kernel(
    input_ptr,
    input_row_stride,
    alpha_ptr,
    gamma_ptr,
    beta_ptr,
    output_ptr,
    output_row_stride,
    num_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Shapes:
        - input: (BT, C)
        - alpha: (1)
        - gamma: (C)
        - beta: (C)
    """
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_cols

    input_ptr += row_idx * input_row_stride
    output_ptr += row_idx * output_row_stride

    alpha = tl.load(alpha_ptr)
    gamma = tl.load(gamma_ptr + offsets, mask=mask)
    beta = tl.load(beta_ptr + offsets, mask=mask)
    input_vals = tl.load(input_ptr + offsets, mask=mask)
    output_vals = gamma * tl.math.tanh((alpha * input_vals).cast(tl.float32)) + beta
    tl.store(output_ptr + offsets, output_vals, mask=mask)


@triton.jit
def dyt_bwd_kernel(
    input_ptr,
    input_row_stride,
    grad_output_ptr,
    grad_output_row_stride,
    grad_input_ptr,
    grad_input_row_stride,
    alpha_ptr,
    grad_alpha_ptr,
    gamma_ptr,
    grad_gamma_ptr,
    grad_gamma_row_stride,
    num_cols,
    num_rows,
    ROWS_PER_PROGRAM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Shapes:
        - input: (BT, C)
        - alpha: (1)
        - gamma: (C)
        - grad_input: (BT, C)
        - grad_output: (BT, C)
        - grad_gamma: (sm_count, C)
        - grad_alpha: (sm_count,)
    """
    pid = tl.program_id(0)

    row_start = pid * ROWS_PER_PROGRAM
    row_end = min((pid + 1) * ROWS_PER_PROGRAM, num_rows)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_cols

    grad_alpha = 0.0
    grad_gamma = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    input_ptr += row_start * input_row_stride
    grad_input_ptr += row_start * grad_input_row_stride
    grad_output_ptr += row_start * grad_output_row_stride
    alpha = tl.load(alpha_ptr)
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=0.0)

    for _ in tl.range(row_start, row_end):
        grad_output_vals = tl.load(grad_output_ptr + offsets, mask=mask, other=0.0)
        input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        tanh_alpha_x = tanh((alpha * input_vals).cast(tl.float32))
        sech2_alpha_x = 1 - tanh_alpha_x * tanh_alpha_x

        grad_input_vals = grad_output_vals * gamma * sech2_alpha_x * alpha
        grad_alpha += tl.sum(grad_output_vals * gamma * sech2_alpha_x * input_vals)
        grad_gamma += grad_output_vals * tanh_alpha_x
        tl.store(grad_input_ptr + offsets, grad_input_vals, mask=mask)

        grad_output_ptr += grad_output_row_stride
        input_ptr += input_row_stride
        grad_input_ptr += grad_input_row_stride

    tl.store(grad_gamma_ptr + pid * grad_gamma_row_stride + offsets, grad_gamma, mask=mask)
    tl.store(grad_alpha_ptr + pid, grad_alpha)

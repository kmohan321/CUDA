import triton
import triton.language as tl

@triton.jit
def rms_norm_forward_kernel(
    Y_ptr, Y_row_stride,
    X_ptr, X_row_stride,
    W_ptr, W_row_stride,
    RSTD_ptr, RSTD_row_stride,
    n_cols, eps, offset: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Compute RMSNorm forward pass:
    y_i = (x_i / RMS) * (offset + w_i), where RMS = sqrt(sum(x_i^2) / N)
    """

    row_id = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_id * Y_row_stride
    X_ptr += row_id * X_row_stride
    RSTD_ptr += row_id * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)

    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    rstd = tl.rsqrt(mean_square + eps)

    tl.store(RSTD_ptr, rstd)

    Y_row = X_row * rstd * (offset + W_row)

    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)


@triton.jit
def rms_norm_backward_kernel(
    dY_ptr, dY_row_stride,
    dX_ptr, dX_row_stride,
    X_ptr, X_row_stride,
    W_ptr, W_row_stride,
    RSTD_ptr, RSTD_row_stride,
    dW_ptr, dW_row_stride,
    n_rows, n_cols,
    offset: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Compute gradients for RMSNorm:
    dx = (1 / RMS) * [dy * (w + offset) - (1 / N) * (1 / RMS^2) * sum(dy * (w + offset) * x) * x]
    dw = sum(dy * (x / RMS))  (summed over BxT dimension)
    """

    row_id = tl.program_id(0)
    row_start = row_id * BLOCK_SIZE
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dW_accum = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    dY_ptr += row_start * dY_row_stride
    dX_ptr += row_start * dX_row_stride
    X_ptr += row_start * X_row_stride
    RSTD_ptr += row_start

    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0) + offset

    for row in range(row_start, min(row_start + BLOCK_SIZE, n_rows)):
        dY_row = tl.load(dY_ptr + col_offsets, mask=mask, other=0.0)
        X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
        rstd_row = tl.load(RSTD_ptr)
        
        m = dY_row * W_row
        dX_row = rstd_row * m

        scale_factor = - (1 / n_cols) * rstd_row * rstd_row * tl.sum(m * X_row, axis=0)
        dX_row += scale_factor * X_row

        dW_accum += dY_row * (X_row * rstd_row)

        tl.store(dX_ptr + col_offsets, dX_row, mask=mask)

        dY_ptr += dY_row_stride
        dX_ptr += dX_row_stride
        X_ptr += X_row_stride
        RSTD_ptr += RSTD_row_stride

    tl.store(dW_ptr + row_id * dW_row_stride + col_offsets, dW_accum, mask=mask)



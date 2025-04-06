import triton
import triton.language as tl 


@triton.jit
def _tv_distance_kernel(
    p_ptr,
    p_stride,
    q_ptr,
    q_stride,
    loss_ptr,
    loss_stride,
    grads_ptr,
    grads_stride,
    label_ptr,
    ignore_index: tl.constexpr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    HAS_LABEL: tl.constexpr,
    reduction: tl.constexpr = _REDUCTION_MODE_BATCHMEAN,
):
    pid = tl.program_id(0).to(tl.int64)
    p_ptr += pid * p_stride
    q_ptr += pid * q_stride
    loss_ptr += pid * loss_stride
    grads_ptr += pid * grads_stride
    label_ptr += pid

    base_offsets = tl.arange(0, BLOCK_SIZE)

    if HAS_LABEL:
        label = tl.load(label_ptr)
        if label == ignore_index:
            for i in range(0, n_cols, BLOCK_SIZE):
                offsets = i + base_offsets
                mask = offsets < n_cols
                tl.store(grads_ptr + offsets, 0.0, mask=mask)
                if reduction == _REDUCTION_MODE_NONE:
                    tl.store(loss_ptr + offsets, 0.0, mask=mask)
            return

    loss_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + base_offsets
        mask = offsets < n_cols

        p = tl.load(p_ptr + offsets, mask=mask, other=0.0)
        q = tl.load(q_ptr + offsets, mask=mask, other=0.0)

        # TVD(P || Q) = 0.5 * |P - Q|
        tv_loss = 0.5 * tl.abs(p - q)

        grad_res = tl.where(p > q, 0.5, -0.5)

        tl.store(grads_ptr + offsets, grad_res, mask=mask)

        if reduction == _REDUCTION_MODE_NONE:
            tl.store(loss_ptr + offsets, tv_loss, mask=mask)
        else:
            loss_sum += tl.sum(tv_loss, axis=0)

    if reduction != _REDUCTION_MODE_NONE:
        tl.store(loss_ptr, loss_sum)
import triton 
import triton.language as tl 

@triton.jit
def embedding_forward_kernel(
    embeddings_ptr,
    indices_ptr,
    output_ptr,
    n_elements,
    embedding_dim: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    start_m = pid_m * BLOCK_SIZE_M # tokens ids are divided in blocks
    start_n = pid_n * BLOCK_SIZE_N # so it is d dimensional , we are loading a part of it 
    offsets_m = start_m + tl.arange(0, BLOCK_SIZE_M) # which token ids block we are working on
    mask_m = offsets_m < n_elements # let say we have n tokens available, think like oupput will (n,d)
    indices = tl.load(indices_ptr + offsets_m, mask=mask_m, other=0) # so we get block_size_m token ids loaded
    offsets_n = start_n + tl.arange(0, BLOCK_SIZE_N) #simply loading blocksize_n dimensions
    mask_n = offsets_n < embedding_dim #not greater than d 

    # shape -> (blocksize_m,blocksize_n)
    #so what we are doing here is that we have some row indices like [2,5,3] etc
    # by multiplying with embedding we will reach at that particular row 
    embedding_offsets = indices[:, None] * embedding_dim + offsets_n[None, :] 
    embeddings = tl.load(
        embeddings_ptr + embedding_offsets,
        mask=mask_m[:, None] & mask_n[None, :],
        other=0.0,
    )
    output_offsets = offsets_m[:, None] * embedding_dim + offsets_n[None, :]
    tl.store(output_ptr + output_offsets, embeddings, mask=mask_m[:, None] & mask_n[None, :])
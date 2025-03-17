import triton
import triton.language as tl 

@triton.jit
def flash_attn(
  Query_ptr,
  Key_ptr,
  Value_ptr,
  b,s,h,d,
  stride_batch_query,
  stride_head_query,
  stride_sequence_query,
  stride_batch_kv,
  stride_head_kv,
  stride_sequence_kv,
  Br:tl.constexpr,
  Bc:tl.constexpr
  
):
  
  batch_head = tl.program_id(0) #giving id of batch_head pair
  query_block = tl.program_id(1) #which query block to load
  
  query_block_offset = query_block * Br
  
  #let's calculate the offsets first 
  batch = batch_head // h 
  head = batch_head % h 
  
  #so we are the right block starting point 
  query_offset = batch * stride_batch_query + head * stride_head_query 
  key_offset = batch * stride_batch_kv + head * stride_head_kv 
  value_offset = batch * stride_batch_kv + head * stride_head_kv 
  
  #let's make the block_ptr for memory access
  query_ptr = tl.make_block_ptr(
    Query_ptr + query_offset,
    [s,d],
    [d,1],
    [query_block_offset,0],
    [Br,d],
    [0,1]
  )
  
  key_ptr = 
  
  
  
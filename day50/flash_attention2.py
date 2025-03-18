import torch
import triton
import triton.language as tl 

@triton.jit
def flash_attn(
  Query_ptr,
  Key_ptr,
  Value_ptr,
  Output_ptr,
  stride_batch_query,
  stride_head_query,
  stride_sequence_query,
  stride_batch_kv,
  stride_head_kv,
  stride_sequence_kv,
  Br:tl.constexpr,
  Bc:tl.constexpr,
  b : tl.constexpr,
  s : tl.constexpr,
  h : tl.constexpr,
  d : tl.constexpr,
  scale
):
  
  batch_head = tl.program_id(0) #giving id of batch_head pair
  query_block = tl.program_id(1) #which query block to load
  
  query_block_offset = query_block * Br
  
  #let's calculate the offsets first 
  batch = batch_head // h 
  head = batch_head % h 
  
  #so we are the right starting point 
  query_offset = batch * stride_batch_query + head * stride_head_query 
  key_offset = batch * stride_batch_kv + head * stride_head_kv 
  value_offset = batch * stride_batch_kv + head * stride_head_kv 
  output_offset = batch * stride_batch_query + head * stride_head_query 
  
  #let's make the block_ptr for memory access
  query_ptr = tl.make_block_ptr(
    Query_ptr + query_offset,
    [s,d],
    [stride_sequence_query,1],
    [query_block_offset,0],
    [Br,d],
    [0,1]
  )
  
  #understand the correct way of tranposing the matrix using the block_ptr
  key_ptr = tl.make_block_ptr(
    Key_ptr + key_offset,
    [d,s], #transposed shape
    [1,stride_sequence_kv],
    [0,0],
    [d,Bc], #transposed block
    [1,0]
  )
  value_ptr = tl.make_block_ptr(
    Value_ptr + value_offset,
    [s,d],
    [stride_sequence_kv,1],
    [0,0],
    [Bc,d],
    [0,1]
  )
  output_ptr = tl.make_block_ptr(
    Output_ptr + output_offset,
    [s,d],
    [stride_sequence_query,1],
    [query_block_offset,0],
    [Br,d],
    [0,1]
  )
  
  #loading query tile
  query_block = tl.load(query_ptr) #shape -> (Br,d)
  
  #intializing the local max and local norm
  local_max = tl.full((Br,),-float("inf"),tl.float32)
  local_norm = tl.zeros((Br,),tl.float32)
  out_acc = tl.zeros((Br,d),tl.float32)
  
  #loading the first block
  key_block = tl.load(key_ptr)
  value_block = tl.load(value_ptr)
  
  for j in range(0,s,Bc):

    key_block = tl.load(key_ptr)
    value_block = tl.load(value_ptr)
    
    #taking the dot product 
    accumalator = tl.dot(query_block,key_block)
    accumalator *= scale 
    
    max_value = tl.max(accumalator,axis=1) #shape -> (Br,)
    block_max = tl.maximum(local_max,max_value) #shape -> (Br,)
    P = tl.math.exp(accumalator - block_max[:,None]) #shape -> (Br,Bc)
    alpha = tl.math.exp(local_max - block_max) #shape -> (Br,)
    local_norm = local_norm * alpha + tl.sum(P,axis=1) #shape -> (Br,)
    
    #adjusting the output
    out_acc = alpha[:,None] * out_acc 
    out_acc += tl.dot(P,value_block)
    local_max = block_max
    
    key_ptr = tl.advance(key_ptr,(0,Bc))
    value_ptr = tl.advance(value_ptr,(Bc,0))
  
  out_acc = out_acc / local_norm[:,None]
  tl.store(output_ptr,out_acc)
  
def flash_attn_pytorch(Q, K, V):
    """Reference PyTorch implementation of Flash Attention"""
    scale = 1/(Q.shape[-1]**0.5)
    attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(attn_weights,dim=-1)
    output = torch.matmul(attn_weights, V)
    return output

def call_flash_attn(batch, seq_len, heads, dim, block_size_r=32, block_size_c=32):
    """Function to allocate memory, launch the Triton kernel, and verify correctness"""
    
    Q = torch.empty(batch, heads, seq_len, dim, dtype=torch.float32, device='cuda').normal_(0,0.5)
    K = torch.empty(batch, heads, seq_len, dim, dtype=torch.float32, device='cuda').normal_(0,0.5)
    V = torch.empty(batch, heads, seq_len, dim, dtype=torch.float32, device='cuda').normal_(0,0.5)
    
    scale = 1.0 / (dim ** 0.5)
    Output = torch.empty_like(Q)
    
    stride_batch_query = Q.stride(0)
    stride_head_query = Q.stride(1)
    stride_sequence_query = Q.stride(2)
    stride_batch_kv = K.stride(0)
    stride_head_kv = K.stride(1)
    stride_sequence_kv = K.stride(2)
    
    num_query_blocks = (seq_len + block_size_r - 1) // block_size_r
    print(f"Launching grid: ({batch * heads}, {num_query_blocks})")
    grid = (batch * heads, num_query_blocks)

    flash_attn[grid](
        Q, K, V, Output,
        stride_batch_query, stride_head_query, stride_sequence_query,
        stride_batch_kv, stride_head_kv, stride_sequence_kv,
        Br=block_size_r, Bc=block_size_c,b = batch, s=seq_len, h=heads,
        d =dim,scale = scale
    )
    
    output_ref = flash_attn_pytorch(Q, K, V)
    # diff = (output_ref - Output).abs()
    # print(f"Max difference: {diff.max().item()}")
    # print(f"Min difference: {diff.min().item()}")
    # print(f"Mean difference: {diff.mean().item()}")
    # print("Sample differences:", diff.flatten()[:10])

    if torch.allclose(Output, output_ref, atol=1e-2):
        print("Triton Flash Attention matches PyTorch implementation!")
    else:
        print("Discrepancy found between Triton and PyTorch implementations.")

call_flash_attn(batch=8, seq_len=1024, heads=16, dim=64)


  
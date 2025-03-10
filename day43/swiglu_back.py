import torch
import triton
import triton.language as tl 

# da(derivative of out with respect to input a ) => b * (sig(a) + swish(a) * (1-sig(a)))
# db(derivative of out with respect to input b) => swish(a)
# dl(derivative of loss with respect to out) 
@triton.jit
def sigmoid(a):
  return a * sigmoid(a)

@triton.jit
def swiglu_backward(
  dL : torch.Tensor,
  a : torch.Tensor,
  b : torch.Tensor,
  stride : int,
  N : tl.constexpr,
  BLOCKSIZE : tl.constexpr
):
  
  row = tl.program_id(0)
  
  #moving the pointers to the exact row 
  dl += row * stride
  da += row * stride
  db += row * stride
  
  n_off = tl.arange(0,BLOCKSIZE)
  mask = n_off < N
  a_row = tl.load(a + n_off, mask=mask,other=0)
  b_row = tl.load(b + n_off, mask=mask,other=0)
  l_row = tl.load(dl + n_off, mask=mask, other=0)
  
  #gradient calculation
  sig = sigmoid(a)
  swish = a * sig
  da = b_row * (sig + swish + (1-sig)) * l_row
  db = swish * l_row
  tl.store(a+n_off, da, mask=mask)
  tl.store(b+n_off, db, mask=mask)
  

  
  
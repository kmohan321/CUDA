import torch
import triton
import triton.language as tl

# swiglu => swish(a) * b => a * sigmoid(a) * b => a * 1/(1 + exp(-a)) * b
# it is a elementwise operation
# going to load whole the row inside one block
@triton.jit
def swish(a):
  return a * tl.sigmoid(a)
  
@triton.jit
def swiglu_triton(
  X1 : torch.Tensor,
  X2 : torch.Tensor,
  Y : torch.Tensor,
  stride_x1 : int,
  stride_x2 : int,
  stride_y : int,
  N : int,
  BlockSize : tl.constexpr

):
  blockid  = tl.program_id(0) #current row
  #moving the pointers
  X1 += blockid * stride_x1
  X2 += blockid * stride_x2
  Y += blockid * stride_y
  
  for i in range(0,N,BlockSize):
    n_off = tl.arange(0,BlockSize) + i
    a = tl.load(X1 + n_off, mask= n_off < N,other = 0)
    b = tl.load(X2 + n_off, mask= n_off < N,other = 0)
    c = swish(a) * b
    tl.store(Y+n_off,c,mask=n_off < N)
    
def swiglu_forward(X1,X2):
  blocksize = 256
  
  #something to remember 
  #shape can be anything like -> (B,s,d) or (M,N)
  o_shape = X1.shape
  X1 = X1.view(-1,o_shape[-1])
  X2 = X2.view(-1, o_shape[-1])
  Y = torch.zeros_like(X1,device='cuda')
  rows = X1.shape[0]
  swiglu_triton[rows,](X1,X2,Y,X1.stride(0),X2.stride(0),Y.stride(0),o_shape[-1],blocksize)
  return Y.view(*o_shape)

def torch_swiglu(x1,x2):
  swish = x1 * torch.sigmoid(x1)
  y = x2 * swish
  return y
  
def benchmark():
  device = 'cuda'
  M,N = 64,256
  x1 = torch.rand(size=(M,N),device=device)
  x2 = torch.rand(size=(M,N),device=device)
  y = swiglu_forward(x1,x2)
  y_torch = torch_swiglu(x1,x2)
  correctness = torch.allclose(y,y_torch,atol=1e-5)
  print(correctness)
  
if __name__ == '__main__':
  benchmark()
  

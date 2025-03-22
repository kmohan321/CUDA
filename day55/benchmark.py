import os
import time
import torch
import math
from torch.nn import functional as F
from torch.utils.cpp_extension import load

# Set CUDA architecture
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

# Load CUDA kernel
flash_attn = load(
    name='flash_attn',
    sources=['wrapper.cpp', 'flash-2.cu'],
    extra_cuda_cflags=['-O3']
)

# Model parameters
batch_size = 16
n_head =   8
seq_len = 512
head_dim = 64

# Create random input tensors
q = torch.randn((batch_size, n_head, seq_len, head_dim), device="cuda")
k = torch.randn((batch_size, n_head, seq_len, head_dim), device="cuda")
v = torch.randn((batch_size, n_head, seq_len, head_dim), device="cuda")

def manual_attn(q, k, v):
    """Standard attention implementation for comparison"""
    scale = 1.0 / math.sqrt(k.size(-1))
    att = (q @ k.transpose(-2, -1)) * scale
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y

def benchmark(func, *args, name="Function"):
    """Measure execution time of a function"""
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    result = func(*args)
    end.record()
    
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end)
    print(f"{name} execution time: {elapsed_time:.3f} ms")
    return result, elapsed_time

def main():
    print(f"Configuration:")
    print(f"Batch size: {batch_size}")
    print(f"Number of heads: {n_head}")
    print(f"Sequence length: {seq_len}")
    print(f"Head dimension: {head_dim}")
    print("\n")

    # Benchmark manual attention
    print("Manual Attention Time")
    manual_result, manual_time = benchmark(manual_attn, q, k, v, name="Manual Attention")

    # Benchmark flash attention implementation
    print("\nFlash Attention Time")
    flash_result, flash_time = benchmark(flash_attn.flash_attn2_forward, q, k, v, name="Flash Attention")

    # Calculate and print speedup
    speedup = manual_time / flash_time
    print(f"\nFlash Attention is {speedup:.2f}x faster than Manual Attention.")

    # Check for numerical accuracy
    print("\n=== Accuracy Check ===")
    tolerance = 1e-2
    is_close = torch.allclose(flash_result, manual_result, rtol=0, atol=tolerance)
    print(f"Results match within tolerance ({tolerance}): {is_close}")

    # Analyze differences if results don't match
    if not is_close:
        diff = torch.abs(flash_result - manual_result)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        print(f"\nMaximum difference: {max_diff:.6f}")
        print(f"Mean difference: {mean_diff:.6f}")

        # Print some specific mismatches
        diff_indices = torch.nonzero(diff > tolerance, as_tuple=True)
        if diff_indices[0].numel() > 0:
            print("\nSample mismatches:")
            for idx in zip(*[d[:4] for d in diff_indices]):  # Show first 4 mismatches
                print(f"Index {idx}:")
                print(f"  Manual: {manual_result[idx].item():.6f}")
                print(f"  Flash:  {flash_result[idx].item():.6f}")
                print(f"  Diff:   {diff[idx].item():.6f}")
# def test_attention_components():
#     # Test with identity matrices
#     q_id = torch.eye(4, device="cuda").view(1,1,4,4)
#     k_id = torch.eye(4, device="cuda").view(1,1,4,4)
#     v_id = torch.ones((1,1,4,4), device="cuda")
    
#     # Manual computation steps
#     scale = 1.0 / math.sqrt(q_id.size(-1))
#     qk = q_id @ k_id.transpose(-2, -1)
#     print("QK matrix:")
#     print(qk[0,0])
    
#     scaled_qk = qk * scale
#     print("\nScaled QK matrix:")
#     print(scaled_qk[0,0])
    
#     attention = torch.softmax(scaled_qk, dim=-1)
#     print("\nAttention weights:")
#     print(attention[0,0])
    
#     # Flash attention
#     flash_result = flash_attn.flash_attn2_forward(q_id, k_id, v_id)
#     print("\nFlash attention result:")
#     print(flash_result[0,0])

# # test_attention_components()
if __name__ == "__main__":
    main()
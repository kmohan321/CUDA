import torch

# Create input tensor
N, C = 1024, 65536  # Batch size, number of classes (adjust as needed)
x = torch.randn(N, C, device="cuda", dtype=torch.float32)

# CUDA events for timing
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Warm-up to stabilize performance
for _ in range(10):
    torch.nn.functional.softmax(x, dim=1)

# Measure PyTorch softmax time
start.record()
y_torch = torch.nn.functional.softmax(x, dim=1)
end.record()

torch.cuda.synchronize()  # Ensure completion
print(f"PyTorch Softmax Time: {start.elapsed_time(end)} ms")

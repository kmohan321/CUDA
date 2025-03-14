import torch
import numpy as np
import ctypes
import os
from PIL import Image
from torchvision import transforms

# Load and preprocess the image
img = Image.open('p1/robot-bird-with-exoskeleton-armor-generative-ai_7023-206712.jpg')
img_tensor = transforms.ToTensor()(img)
img_tensor = img_tensor.permute(1,2,0)
img_tensor = img_tensor.cpu().numpy()  # Convert to numpy array

print(img_tensor.shape)
height, width,_ = img_tensor.shape  # Note the change in dimension order

# Prepare output array
output = np.zeros((height, width), dtype=np.float32)

# Load the CUDA shared library
lib_path = r'C:\Users\Krishna Mohan\OneDrive\Desktop\CUDA_p\1.dll'
try:
    cuda_lib = ctypes.CDLL(lib_path)
    print(f"Successfully loaded the library from {lib_path}")
except OSError as e:
    print(f"Failed to load the library from {lib_path}")
    print(f"Error: {e}")
    raise

# List all functions in the DLL
print("Functions available in the DLL:")
for func_name, func in cuda_lib.__dict__.items():
    if isinstance(func, ctypes._CFuncPtr):
        print(f" - {func_name}")

# Try to get the function
try:
    launch_kernel = cuda_lib.main
    print("Successfully found the launch_kernel function")
except AttributeError:
    print("launch_kernel function not found. Please check the function name in your CUDA code.")
    raise

# Define argument types for the C function
launch_kernel.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    ctypes.c_int
]

# Flatten the input and output arrays
img_tensor_flat = img_tensor.ravel().astype(np.float32)
output_flat = output.ravel().astype(np.float32)

# Call the CUDA function
launch_kernel(
    img_tensor_flat,
    output_flat,
    width,
    height
)

# Reshape the output back to 2D
output = output_flat.reshape((height, width))

# Convert back to PyTorch tensor if needed
output_tensor = torch.from_numpy(output).to('cuda')

print("Processing complete. Output shape:", output_tensor.shape)
from torchvision.utils import save_image
save_image(output_tensor,'1.png')
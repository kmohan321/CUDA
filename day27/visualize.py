import numpy as np
import matplotlib.pyplot as plt
import cv2

h, w, C = 100, 100, 3

# Read raw images
x_t_1 = np.fromfile("output.raw", dtype=np.float32).reshape(h, w, C)
noise = np.fromfile("noise.raw", dtype=np.float32).reshape(h, w, C)

# Normalize both for visualization
x_t_1 = (x_t_1 * 255).astype(np.uint8)
noise_vis = (noise - noise.min()) / (noise.max() - noise.min())  # Normalize noise [0,1]
noise_vis = (noise_vis * 255).astype(np.uint8)  # Scale to [0,255]

# Convert to BGR for OpenCV
x_t_1_bgr = cv2.cvtColor(x_t_1, cv2.COLOR_RGB2BGR)

# Show images
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(x_t_1_bgr)
ax[0].set_title("Noised Image (x_t_1)")
ax[0].axis("off")

ax[1].imshow(noise_vis, cmap="magma")  # Heatmap for noise
ax[1].set_title("Noise Heatmap")
ax[1].axis("off")

plt.show()

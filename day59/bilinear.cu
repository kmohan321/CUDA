#include<stdio.h>
#include<cuda.h>

__global__ void bilinear_interpolation(const float* input, float* output, int in_width, int in_height, int out_width, int out_height) {
  int x_out = blockIdx.x * blockDim.x + threadIdx.x;
  int y_out = blockIdx.y * blockDim.y + threadIdx.y;

  if (x_out >= out_width || y_out >= out_height) return;

  // Compute input coordinates
  float x_in = (x_out + 0.5f) * (float(in_width) / out_width) - 0.5f;
  float y_in = (y_out + 0.5f) * (float(in_height) / out_height) - 0.5f;

  int x1 = floorf(x_in);
  int y1 = floorf(y_in);
  int x2 = min(x1 + 1, in_width - 1);
  int y2 = min(y1 + 1, in_height - 1);

  float dx = x_in - x1;
  float dy = y_in - y1;

  // Load four neighbors
  float q11 = input[y1 * in_width + x1];
  float q21 = input[y1 * in_width + x2];
  float q12 = input[y2 * in_width + x1];
  float q22 = input[y2 * in_width + x2];

  // Compute interpolation
  float interpolated_value = q11 * (1 - dx) * (1 - dy) +
                             q21 * dx * (1 - dy) +
                             q12 * (1 - dx) * dy +
                             q22 * dx * dy;

  output[y_out * out_width + x_out] = interpolated_value;
}

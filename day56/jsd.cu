#include<stdio.h>
#include<cuda.h>
#include<torch/types.h>


__global__ void jsd_kernel(
  const float* __restrict__ X, // input in logspace, X = log Q
  const float* __restrict__ Y, // ground truth in logspace, Y = log P
  float* __restrict__ loss,
  float* __restrict__ dX,
  const int* __restrict__ labels,
  float beta,
  int n_non_ignore,
  int ignore_index,
  int n_cols
) {
  int row = blockIdx.x; // Each block processes one row
  int col = threadIdx.x + blockDim.x * blockIdx.y; // Columns are processed by threads
  
  if (col >= n_cols) return;
  
  int index = row * n_cols + col;
  float x = X[index];
  float y = Y[index];
  float dX_val = 0.0f;
  float loss_val = 0.0f;
  
  if (labels && labels[row] == ignore_index) {
      dX[index] = 0.0f;
      return;
  }
  
  if (beta == 0.0f) { // Forward KL
      float y_prob = expf(y);
      loss_val = y_prob * (y - x);
      dX_val = -y_prob;
  } else if (beta == 1.0f) { // Reverse KL
      float x_prob = expf(x);
      loss_val = x_prob * (x - y);
      dX_val = loss_val + x_prob;
  } else { // JSD or Generalized KL
      float q = expf(x);
      float p = expf(y);
      float M = beta * p + (1 - beta) * q;
      float log_M = logf(M);
      loss_val = beta * p * y + (1 - beta) * q * x - M * log_M;
      dX_val = (1 - beta) * q * (x - log_M);
  }
  
  float scale = 1.0f / n_non_ignore;
  loss[index] = loss_val * scale;
  dX[index] = dX_val * scale;
}

torch::Tensor jsd_forward(
  torch::Tensor X, torch::Tensor Y, torch::Tensor loss, torch::Tensor dX,
  torch::Tensor labels, float beta, int ignore_index, int has_label
) {
  int BT = X.size(0), V = X.size(1);
  int n_non_ignore = has_label ? (labels != ignore_index).sum().item<int>() : BT;
  
  dim3 blockDim(256);
  dim3 gridDim(BT, (V + blockDim.x - 1) / blockDim.x);
  
  jsd_kernel<<<gridDim, blockDim>>>(
      X.data_ptr<float>(), Y.data_ptr<float>(), loss.data_ptr<float>(),
      dX.data_ptr<float>(), has_label ? labels.data_ptr<int>() : nullptr,
      beta, n_non_ignore, ignore_index, V
  );
}

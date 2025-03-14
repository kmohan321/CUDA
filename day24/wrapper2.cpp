#include <torch/extension.h>
#include <vector>

// Declaration of the CUDA kernel
torch::Tensor fa_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fa_forward", &fa_forward, "Flash Attention forward");
}
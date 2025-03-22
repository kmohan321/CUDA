#include <torch/extension.h>

torch::Tensor flash_attn2_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V );
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_attn2_forward", &flash_attn2_forward, "Flash Attention 2 forward");
}


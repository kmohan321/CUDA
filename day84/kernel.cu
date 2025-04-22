#include <cutlass/cutlass.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/conv/kernel/default_conv2d_fprop.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>

using ElementInput = float;
using ElementOutput = float;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutInput = cutlass::layout::TensorNHWC;
using LayoutFilter = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

using Conv2dFpropKernel = typename cutlass::conv::kernel::DefaultConv2dFprop<
    ElementInput, LayoutInput,
    ElementInput, LayoutFilter,
    ElementOutput, LayoutOutput,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<8, 8, 4>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        1,
        ElementAccumulator,
        ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    cutlass::arch::OpMultiplyAdd,
    cutlass::conv::IteratorAlgorithm::kAnalytic
>::Kernel;

using Conv2dFprop = cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/gemm.h>
#include <iostream>

int main() {
    using ElementInputA = float;
    using ElementInputB = float;
    using ElementOutput = float;
    using ElementCompute = float;

    constexpr int M = 128, N = 128, K = 128;

    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::RowMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementCompute
    >;

    cutlass::HostTensor<ElementInputA, LayoutInputA> A({M, K});
    cutlass::HostTensor<ElementInputB, LayoutInputB> B({K, N});
    cutlass::HostTensor<ElementOutput, LayoutOutput> C({M, N});

    A.fill_random();
    B.fill_random();
    C.fill(0);

    Gemm gemm_op;
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    Gemm::Arguments args{problem_size,
                         A.device_ref(), B.device_ref(), C.device_ref(), C.device_ref(),
                         {1.0f, 0.0f}};

    cutlass::Status status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM kernel failed\n";
        return -1;
    }

    std::cout << "CUTLASS GEMM completed.\n";
    return 0;
}

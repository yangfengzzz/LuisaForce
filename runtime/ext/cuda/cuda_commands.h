//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/dll_export.h"
#include "runtime/ext/cuda/lcub/dcub_common.h"
#include "runtime/ext/cuda/lcub/cuda_lcub_command.h"

namespace luisa::compute::cuda {
class LC_BACKEND_API CudaCommand {
public:
    template<typename T>
    using BufferView = luisa::compute::BufferView<T>;
    using UCommand = luisa::unique_ptr<luisa::compute::cuda::CudaLCubCommand>;
public:
    static UCommand mad_throughput(BufferView<float> src0_buffer, BufferView<float> src1_buffer, BufferView<float> dst_buffer) noexcept;

    template<uint TILE_M, uint TILE_N, uint TILE_K, uint WG_X, uint WG_Y>
    static UCommand matmul(BufferView<float> src0_buffer, BufferView<float> src1_buffer, BufferView<float> dst_buffer,
                           int M, int N, int K) noexcept;

    // matrix-matrix transposed multiplication of two 2D inputs.
    static UCommand mmt(BufferView<float> src0_buffer, BufferView<float> src1_buffer, BufferView<float> dst_buffer,
                        int tileM, int tileN, int tileK,
                        int M, int N, int K,
                        int wg_size_x, int wg_size_y) noexcept;

    enum class ReduceMode {
        Loop,
        SimdGroup
    };

    static UCommand atomic_reduce(BufferView<float> src_buffer, BufferView<float> dst_buffer,
                                  size_t batch_elements, ReduceMode mode, bool is_integer) noexcept;

    static UCommand one_workgroup_reduce(BufferView<float> src_buffer, BufferView<float> dst_buffer,
                                         size_t batch_elements, ReduceMode mode) noexcept;

    static UCommand tree_reduce(BufferView<float> buffer,
                                size_t batch_elements, ReduceMode mode, bool is_integer) noexcept;

    enum class ArithmeticMode {
        Add,
        Mul
    };

    static UCommand simd_group_arithmetic(BufferView<float> src_buffer, BufferView<float> dst_buffer,
                                          size_t batch_elements, ArithmeticMode mode) noexcept;
};
}// namespace luisa::compute::cuda

#include "custom/matmul.h"
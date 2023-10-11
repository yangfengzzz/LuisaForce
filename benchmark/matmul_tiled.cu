//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#ifdef LUISA_PLATFORM_CUDA
#include "runtime/ext/cuda/cuda_commands.h"
#endif

namespace luisa::compute::cuda {
#define SHADER_TILE_COMPILE(TILE_M, TILE_N, TILE_K, X, Y)                                                           \
    template CudaCommand::UCommand CudaCommand::matmul<TILE_M, TILE_N, TILE_K, X, Y>(BufferView<float> src0_buffer, \
                                                                                     BufferView<float> src1_buffer, \
                                                                                     BufferView<float> dst_buffer,  \
                                                                                     int M, int N, int K) noexcept;

#define WORKGROUP_TILE_N(X, Y, N)       \
    SHADER_TILE_COMPILE(2, N, 4, X, Y)  \
    SHADER_TILE_COMPILE(4, N, 4, X, Y)  \
    SHADER_TILE_COMPILE(8, N, 4, X, Y)  \
    SHADER_TILE_COMPILE(16, N, 4, X, Y) \
    SHADER_TILE_COMPILE(32, N, 4, X, Y) \
    SHADER_TILE_COMPILE(2, N, 8, X, Y)  \
    SHADER_TILE_COMPILE(4, N, 8, X, Y)  \
    SHADER_TILE_COMPILE(8, N, 8, X, Y)  \
    SHADER_TILE_COMPILE(16, N, 8, X, Y) \
    SHADER_TILE_COMPILE(32, N, 8, X, Y)

WORKGROUP_TILE_N(16, 1, 64)
WORKGROUP_TILE_N(16, 1, 128)
}// namespace luisa::compute::cuda
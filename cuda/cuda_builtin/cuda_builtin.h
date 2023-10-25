//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#define WP_NO_CRT
#include "stl/hashgrid.h"

#include "cuda_jit.h"

namespace wp {
#ifndef BLOCK_SIZE
#define BLOCK_SIZE vec_t<3, size_t>(16, 16, 16)
#endif

#define dispatch_size() vec_t<3, size_t>(params.ls_kid)

#define kernel_id() static_cast<size_t>(params.ls_kid.w())

[[nodiscard]] CUDA_CALLABLE_DEVICE constexpr vec_t<3, size_t> block_size() noexcept {
    return BLOCK_SIZE;
}

[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto thread_id() noexcept {
#ifdef __CUDACC__
    return vec_t<3, size_t>(size_t(threadIdx.x),
                            size_t(threadIdx.y),
                            size_t(threadIdx.z));
#else
    return s_threadIdx;
#endif
}

[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto block_id() noexcept {
#ifdef __CUDACC__
    return vec_t<3, size_t>(size_t(blockIdx.x),
                            size_t(blockIdx.y),
                            size_t(blockIdx.z));
#else
    return s_threadIdx;
#endif
}

[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto dispatch_id() noexcept {
    return block_size() * block_id() + thread_id();
}

CUDA_CALLABLE_DEVICE inline void synchronize_block() noexcept {
#ifdef __CUDACC__
    __syncthreads();
#endif
}

}// namespace wp

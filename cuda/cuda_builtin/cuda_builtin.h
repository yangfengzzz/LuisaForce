//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#define WP_NO_CRT
#include "stl/hashgrid.h"
#include "stl/buffer.h"
#include "cuda_jit.h"

#define kernel_id() static_cast<size_t>(params.ls_kid.w())

namespace wp {
#ifndef BLOCK_SIZE
#define BLOCK_SIZE vec_t<3, wp_uint>(16, 16, 16)
#endif

#define dispatch_size() vec_t<3, wp::wp_uint>(params.ls_kid)

[[nodiscard]] CUDA_CALLABLE_DEVICE constexpr vec_t<3, wp_uint> block_size() noexcept {
    return BLOCK_SIZE;
}

[[nodiscard]] CUDA_CALLABLE_DEVICE inline wp_uint3 thread_id() noexcept {
#ifdef __CUDACC__
    return vec_t<3, wp_uint>(wp_uint(threadIdx.x),
                             wp_uint(threadIdx.y),
                             wp_uint(threadIdx.z));
#else
    return s_threadIdx;
#endif
}

[[nodiscard]] CUDA_CALLABLE_DEVICE inline wp_uint3 block_id() noexcept {
#ifdef __CUDACC__
    return vec_t<3, wp_uint>(wp_uint(blockIdx.x),
                             wp_uint(blockIdx.y),
                             wp_uint(blockIdx.z));
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

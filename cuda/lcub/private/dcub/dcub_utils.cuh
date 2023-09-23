//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "runtime/ext/cuda/lcub/dcub_common.h"
#include <cub/thread/thread_operators.cuh>

namespace luisa::compute::cuda::dcub {
template<typename F>
inline cudaError_t op_mapper(BinaryOperator op, F &&f) noexcept {
    switch (op) {
        case BinaryOperator::Max:
            return f(cub::Max{});
        case BinaryOperator::Min:
            return f(cub::Min{});
        default:
            return f(cub::Max{});
    }
}

struct Difference {
    template<typename T>
    __host__ __device__
        __forceinline__ T
        operator()(const T &lhs, const T &rhs) const noexcept { return lhs - rhs; }
};
}// namespace luisa::compute::cuda::dcub
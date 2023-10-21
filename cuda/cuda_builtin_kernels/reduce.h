//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma

#include "cuda_builtin/math/cuda_math_utils.h"
#include <cuda.h>

namespace luisa::compute::cuda {
void array_inner_float_host(uint64_t a, uint64_t b, uint64_t out, int count, int stride_a, int stride_b, int type_len);
void array_inner_double_host(uint64_t a, uint64_t b, uint64_t out, int count, int stride_a, int stride_b, int type_len);
void array_inner_float_device(uint64_t a, uint64_t b, uint64_t out, int count, int stride_a, int stride_b, int type_len, CUstream stream);
void array_inner_double_device(uint64_t a, uint64_t b, uint64_t out, int count, int stride_a, int stride_b, int type_len, CUstream stream);

void array_sum_float_device(uint64_t a, uint64_t out, int count, int stride, int type_len, CUstream stream);
void array_sum_float_host(uint64_t a, uint64_t out, int count, int stride, int type_len);
void array_sum_double_host(uint64_t a, uint64_t out, int count, int stride, int type_len);
void array_sum_double_device(uint64_t a, uint64_t out, int count, int stride, int type_len, CUstream stream);

}// namespace luisa::compute::cuda
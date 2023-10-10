//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma

#include "cuda/math/cuda_math_utils.h"

namespace luisa::compute::cuda {

void runlength_encode_int_device(uint64_t values, uint64_t run_values, uint64_t run_lengths, uint64_t run_count, int n);

}// namespace luisa::compute::cuda
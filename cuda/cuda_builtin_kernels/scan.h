//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace luisa::compute::cuda {

template<typename T>
void scan_device(const T *values_in, T *values_out, int n, bool inclusive, CUstream stream);

void array_scan_int_device(uint64_t in, uint64_t out, int len, bool inclusive, CUstream stream);

void array_scan_float_device(uint64_t in, uint64_t out, int len, bool inclusive, CUstream stream);

}// namespace luisa::compute::cuda

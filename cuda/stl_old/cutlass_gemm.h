//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma

#include "cuda/math/cuda_math_utils.h"

namespace luisa::compute::cuda {

bool cutlass_gemm(int compute_capability, int m, int n, int k, const char *datatype,
                  const void *a, const void *b, const void *c, void *d, float alpha, float beta,
                  bool row_major_a, bool row_major_b, bool allow_tf32x3_arith, int batch_count);

}// namespace luisa::compute::cuda
//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <metal_stdlib>
using namespace metal;

constant int kArraySize [[function_constant(0)]];

uint count [[simdgroups_per_threadgroup]];
uint index [[thread_position_in_grid]];

#ifndef ARITHMETIC_ADD
#ifndef ARITHMETIC_MUL
#define ARITHMETIC_MUL
#endif
#endif

kernel void simdgroup_arithmetic_intrinsic(device float* input_values [[buffer(0)]],
                                           device float* output_values [[buffer(1)]],
                                           uint3 tpig [[ thread_position_in_grid ]]) {
    float value = 0.f;
    
#ifdef ARITHMETIC_ADD
    float total = simd_sum(input_values[index]);
#endif
#ifdef ARITHMETIC_MUL
    float total = simd_product(input_values[index]);
#endif
    if (simd_is_first()) {
        value = total;
    } else {
        value = input_values[index];
    }
    
    output_values[index] = value;
}

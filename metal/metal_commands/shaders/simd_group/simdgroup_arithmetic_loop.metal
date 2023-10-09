//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <metal_stdlib>
using namespace metal;

#ifndef ARITHMETIC_ADD
#ifndef ARITHMETIC_MUL
#define ARITHMETIC_MUL
#endif
#endif

kernel void simdgroup_arithmetic_loop(device float* input_values [[buffer(0)]],
                                      device float* output_values [[buffer(1)]],
                                      uint count [[threads_per_simdgroup]],
                                      uint index [[thread_position_in_grid]]) {
    float value = 0.f;
    
    if (simd_is_first()) {
#ifdef ARITHMETIC_ADD
        value = 0.0f;
        for (uint i = 0; i < count; ++i)
            value += input_values[index + i];
#endif
        
#ifdef ARITHMETIC_MUL
        value = 1.0f;
        for (uint i = 0; i < count; ++i)
            value *= input_values[index + i];
#endif
    } else {
        value = input_values[index];
    }
    
    output_values[index] = value;
}

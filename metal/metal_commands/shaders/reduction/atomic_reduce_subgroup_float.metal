//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <metal_stdlib>
using namespace metal;

#ifndef BATCH_SIZE
#define BATCH_SIZE 8
#endif

kernel void atomic_reduce_subgroup_float(device float4* Input [[buffer(0)]],
                                         device atomic<float>* Output [[buffer(1)]],
                                         uint wgID [[threadgroup_position_in_grid]],
                                         uint laneID [[thread_position_in_threadgroup]]) {
    uint wgBaseOffset = wgID * BATCH_SIZE / 4;
    float4 laneResult = Input[wgBaseOffset + laneID];
    
    for (uint i = 1; i < BATCH_SIZE / (16 * 4); ++i) {
        laneResult += Input[wgBaseOffset + 16 * i + laneID];
    }
    
    // Final reduction with one subgroup
    float4 wgResult = simd_sum(laneResult);
    float floatResult = wgResult.x + wgResult.y + wgResult.z + wgResult.w;

    if (simd_is_first()) {
        atomic_fetch_add_explicit(Output, floatResult, memory_order::memory_order_relaxed);
    }
}

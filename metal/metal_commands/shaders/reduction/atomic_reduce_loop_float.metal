//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <metal_stdlib>
using namespace metal;

uint wgID [[threadgroup_position_in_grid]];
uint laneID [[thread_position_in_threadgroup]];

constant uint BATCH_SIZE [[function_constant(0)]];

kernel void atomic_reduce_loop_float(device float4* Input [[buffer(0)]],
                                     device atomic<float>* Output [[buffer(1)]]) {
    if (laneID != 0) return;
    
    uint wgBaseOffset = wgID * BATCH_SIZE / 4;
    float4 wgResult = Input[wgBaseOffset];
    
    for (uint i = 1; i < BATCH_SIZE / 4; ++i) {
        wgResult += Input[wgBaseOffset + i];
    }
    
    int floatResult = wgResult.x + wgResult.y + wgResult.z + wgResult.w;
    atomic_fetch_add_explicit(Output, floatResult, memory_order::memory_order_relaxed);
}


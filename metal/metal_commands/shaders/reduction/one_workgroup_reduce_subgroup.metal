//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <metal_stdlib>
using namespace metal;

uint laneCount [[threadgroups_per_grid]];
uint laneID [[thread_position_in_threadgroup]];

#ifndef totalCount
#define totalCount 8
#endif

kernel void one_workgroup_reduce_subgroup(device float4* Input [[buffer(0)]],
                                          device float* Output [[buffer(1)]]) {
    float4 laneResult = Input[laneID];
    
    uint numBatches = totalCount / (laneCount * 4);
    for (uint i = 1; i < numBatches; ++i) {
        laneResult += Input[laneCount * i + laneID];
    }
    
    // Final reduction with one subgroup
    float4 wgResult = simd_sum(laneResult);
    float floatResult = dot(wgResult, float4(1.f, 1.f, 1.f, 1.f));
    
    if (simd_is_first())
        Output[0] = floatResult;
}

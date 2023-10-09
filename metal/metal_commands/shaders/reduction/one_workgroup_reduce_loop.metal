//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <metal_stdlib>
using namespace metal;

uint laneID [[thread_position_in_threadgroup]];

#ifndef totalCount
#define totalCount 8
#endif

kernel void one_workgroup_reduce_loop(device float4* Input [[buffer(0)]],
                                      device float* Output [[buffer(1)]]) {
    if (laneID != 0) return;
    
    float4 wgResult = Input[0];
    
    for (uint i = 1; i < totalCount / 4; ++i) {
        wgResult += Input[i];
    }
    
    float floatResult = dot(wgResult, float4(1.f, 1.f, 1.f, 1.f));
    
    Output[0] = floatResult;
}

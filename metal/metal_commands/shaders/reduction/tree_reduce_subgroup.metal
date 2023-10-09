//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <metal_stdlib>
using namespace metal;

uint wgID [[threadgroup_position_in_grid]];
uint laneID [[thread_position_in_threadgroup]];
uint laneCount [[threadgroups_per_grid]];

#ifndef stride
#define stride 8
#endif

#ifndef BATCH_SIZE
#define BATCH_SIZE 8
#endif

kernel void tree_reduce_subgroup(device float* IOBuffer [[buffer(0)]]) {
    float laneResult = IOBuffer[wgID + stride * laneID];
    
    for (uint i = 1; i < BATCH_SIZE / 16; ++i) {
        laneResult += IOBuffer[wgID + stride * (laneCount * i + laneID)];
    }
    
    // Final reduction with one subgroup
    float wgResult = simd_sum(laneResult);
    if (simd_is_first()) {
        IOBuffer[wgID] = wgResult;
    }
}


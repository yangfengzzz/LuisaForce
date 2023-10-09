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

#ifndef TYPE
#define TYPE float
#endif

kernel void tree_reduce_loop(device TYPE* IOBuffer [[buffer(0)]],
                             uint wgID [[threadgroup_position_in_grid]],
                             uint laneID [[thread_position_in_threadgroup]],
                             uint stride [[threadgroups_per_grid]]) {
    if (laneID != 0) return;
    
    TYPE wgResult = IOBuffer[wgID];
    
    for (uint i = 1; i < BATCH_SIZE; ++i) {
        wgResult += IOBuffer[wgID + stride * i];
    }
    
    IOBuffer[wgID] = wgResult;
}


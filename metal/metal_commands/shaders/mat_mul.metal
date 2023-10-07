//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <metal_stdlib>
using namespace metal;

uint2 gid [[thread_position_in_grid]];
uint2 grid_size [[threads_per_grid]];

kernel void mat_mul(device float* M [[buffer(0)]],
                    device float* N [[buffer(1)]],
                    device float* P [[buffer(2)]]) {
    uint row = gid.y;
    uint col = gid.x;
    uint width = grid_size.x;
    
    float pValue = 0;
    for (uint k = 0; k < width; k++) {
        pValue += M[row * width + k] * N[k * width + col];
    }
    P[row * width + col] = pValue;
}

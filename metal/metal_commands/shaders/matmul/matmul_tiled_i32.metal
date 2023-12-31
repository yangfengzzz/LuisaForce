//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <metal_stdlib>
using namespace metal;

constant uint M [[function_constant(0)]];
constant uint N [[function_constant(1)]];
constant uint K [[function_constant(2)]];

#ifndef TILE_M
#define TILE_M 8
#endif

#ifndef TILE_N
#define TILE_N 8
#endif

#ifndef TILE_K
#define TILE_K 8
#endif

#ifndef WG_X
#define WG_X 1
#endif

#ifndef WG_Y
#define WG_Y 1
#endif

uint coordToOffset(uint i, uint j, uint stride) {
    return (stride * i + j);
}

kernel void matmul_tiled_i32(device int4* inputA [[buffer(0)]],
                             device int4* inputB [[buffer(1)]],
                             device int4* outputO [[buffer(2)]],
                             uint2 gID [[threadgroup_position_in_grid]],
                             uint2 laneId [[thread_position_in_threadgroup]]) {
    const uint strideA = K;
    const uint strideB = N;
    const uint strideC = N;
    
    constexpr uint C_ROWS = TILE_M / WG_Y;
    constexpr uint C_COLS = TILE_N / (4*WG_X);
    int4 C[C_ROWS][C_COLS];
    int4 B[TILE_K][C_COLS];
    
    // Initialize result to zero
    for (uint i = 0; i < C_ROWS; ++i) {
        for (uint j = 0; j < C_COLS; ++j) {
            C[i][j] = int4(0, 0, 0, 0);
        }
    }
    
    for (uint k = 0; k < K; k += TILE_K) {
        for (uint j = 0; j < C_COLS; ++j) {
            for (uint i = 0; i < TILE_K; ++i) {
                uint gj = gID.x * (TILE_N / 4) + laneId.x + j * WG_X;
                uint gk = k + i;
                B[i][j] = inputB[coordToOffset(gk, gj, strideB / 4)];
            }
        }
        
        for (uint i = 0; i < C_ROWS; ++i) {
            uint gi = gID.y * TILE_M + laneId.y + i * WG_Y;
            uint gk = k / 4;
            for (uint kk = 0; kk < TILE_K / 4; ++kk) {
                int4 A = inputA[coordToOffset(gi, gk + kk, strideA / 4)];
                for (uint j = 0; j < C_COLS; ++j) {
                    C[i][j] += int4(A.x, A.x, A.x, A.x) * (B[0 + 4 * kk][j]);
                    C[i][j] += int4(A.y, A.y, A.y, A.y) * (B[1 + 4 * kk][j]);
                    C[i][j] += int4(A.z, A.z, A.z, A.z) * (B[2 + 4 * kk][j]);
                    C[i][j] += int4(A.w, A.w, A.w, A.w) * (B[3 + 4 * kk][j]);
                }
            }
        }
    }
    
    for (uint i = 0; i < C_ROWS; ++i) {
        for (uint j = 0; j < C_COLS; ++j) {
            uint gi = gID.y * TILE_M + laneId.y + i * WG_Y;
            uint gj = gID.x * (TILE_N / 4) + laneId.x + j * WG_X;
            outputO[gi * strideC / 4 + gj] = C[i][j];
        }
    }
}

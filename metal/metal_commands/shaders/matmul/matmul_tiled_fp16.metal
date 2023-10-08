//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <metal_stdlib>
using namespace metal;

uint2 gID [[threadgroup_position_in_grid]];
uint2 laneId [[thread_position_in_threadgroup]];

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

kernel void matmul_tiled_fp16(device float2* inputA [[buffer(0)]],
                              device float4* inputB [[buffer(1)]],
                              device uint4* outputO [[buffer(2)]],
                              texture2d<float> texB [[texture(0)]]) {
    const uint strideA = K;
    const uint strideB = N;
    const uint strideC = N;
    
    constexpr uint C_ROWS = TILE_M / WG_Y;
    constexpr uint C_COLS = TILE_N / (4*WG_X);
    half4 C[C_ROWS][C_COLS];
    half4 B[TILE_K][C_COLS];
    
    // Initialize result to zero
    for (uint i = 0; i < C_ROWS; ++i) {
        for (uint j = 0; j < C_COLS; ++j) {
            C[i][j] = half4(0.f, 0.f, 0.f, 0.f);
        }
    }
    
    for (uint k = 0; k < K; k+=TILE_K) {
        for (uint j = 0; j < C_COLS; j+=2) {
            for (uint i = 0; i < TILE_K; ++i) {
                uint gj = gID.x * TILE_N/4 + laneId.x*2 + j*WG_X;
                uint gk = k+i;
#if (TEXTURE == 1)
                vec4 temp = texelFetch(texB, ivec2(gj/2, gk), 0);
#else
                float4 temp = inputB[coordToOffset(gk, gj/2, strideB/8)];
#endif
                B[i][j].x = unpack_unorm2x16_to_half(as_type<uint>(temp.x)).x;
                B[i][j].y = unpack_unorm2x16_to_half(as_type<uint>(temp.x)).y;
                B[i][j].z = unpack_unorm2x16_to_half(as_type<uint>(temp.y)).x;
                B[i][j].w = unpack_unorm2x16_to_half(as_type<uint>(temp.y)).y;
                B[i][j+1].x = unpack_unorm2x16_to_half(as_type<uint>(temp.z)).x;
                B[i][j+1].y = unpack_unorm2x16_to_half(as_type<uint>(temp.z)).y;
                B[i][j+1].z = unpack_unorm2x16_to_half(as_type<uint>(temp.w)).x;
                B[i][j+1].w = unpack_unorm2x16_to_half(as_type<uint>(temp.w)).y;
            }
        }
        
        for (uint i = 0; i < C_ROWS; ++i) {
            uint gi = gID.y * TILE_M + laneId.y + i*WG_Y;
            uint gk = k/4;
            for (uint kk = 0; kk < TILE_K/4; kk++) {
                float2 temp = inputA[coordToOffset(gi, gk+kk, strideA/4)];
                half a;
                for (uint j = 0; j < C_COLS; ++j) {
                    a = unpack_unorm2x16_to_half(as_type<uint>(temp.x)).x;
                    C[i][j] += half4(a, a, a, a)*B[0+4*kk][j];
                    a = unpack_unorm2x16_to_half(as_type<uint>(temp.x)).y;
                    C[i][j] += half4(a, a, a, a)*B[1+4*kk][j];
                    a = unpack_unorm2x16_to_half(as_type<uint>(temp.y)).x;
                    C[i][j] += half4(a, a, a, a)*B[2+4*kk][j];
                    a = unpack_unorm2x16_to_half(as_type<uint>(temp.y)).y;
                    C[i][j] += half4(a, a, a, a)*B[3+4*kk][j];
                }
            }
        }
    }
    
    for (uint i = 0; i < C_ROWS; ++i) {
        for (uint j = 0; j < C_COLS; j+=2) {
            uint gi = gID.y * TILE_M + laneId.y + i*WG_Y;
            uint gj = gID.x * TILE_N/4 + laneId.x*2 + j*WG_X;
            uint4 temp;
            temp.x = pack_half_to_unorm2x16(half2(C[i][j].xy));
            temp.y = pack_half_to_unorm2x16(half2(C[i][j].zw));
            temp.z = pack_half_to_unorm2x16(half2(C[i][j+1].xy));
            temp.w = pack_half_to_unorm2x16(half2(C[i][j+1].zw));
            outputO[gi * strideC/8 + gj/2] = temp;
        }
    }
}

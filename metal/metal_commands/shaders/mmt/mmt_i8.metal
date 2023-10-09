//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <metal_stdlib>
using namespace metal;

#ifndef M
#define M 8
#endif

#ifndef N
#define N 8
#endif

#ifndef K
#define K 8
#endif

#ifndef M0
#define M0 8
#endif

#ifndef N0
#define N0 8
#endif

#ifndef K0
#define K0 8
#endif

#ifndef WG_X
#define WG_X 1
#endif

#ifndef WG_Y
#define WG_Y 1
#endif

/// Returns the index of `X[i, j]`, where `X` is a 2D matrix of stride |stride|.
uint coordToOffset(uint i, uint j, uint stride) {
    return stride * i + j;
}

int32_t sdot(char4 lhs, char4 rhs) {
    short4 mul = short4(lhs) * short4(rhs);
    return int32_t(mul.x) + int32_t(mul.y) + int32_t(mul.z) + int32_t(mul.w);
}

kernel void mmt_i8(device char4* inputA [[buffer(0)]],
                   device char4* inputB [[buffer(1)]],
                   device int32_t* outputO [[buffer(2)]],
                   uint2 wgID [[threadgroup_position_in_grid]],
                   uint2 localID [[thread_position_in_threadgroup]],
                   uint threadID [[thread_index_in_simdgroup]]) {
    constexpr uint VECTORIZE_K = 4;
    const uint K_VEC = K / VECTORIZE_K;
    constexpr uint K0_VEC = K0 / VECTORIZE_K;
        
    const uint strideA = K_VEC; // Stride of the `inputA` matrix.
    const uint strideB = K_VEC; // Stride of the `inputB` matrix.
    const uint strideC = N; // Stride of the `outputO` matrix.
    
    // Each workgroup processes an output tile of size [M0 x N0], therefore
    // each thread processes a [M0/WG_Y x N0/WG_X] subview.
    constexpr uint C_ROWS = M0 / WG_Y;
    constexpr uint C_COLS = N0 / WG_X;
    
    // The start offsets of the tile processed by this thread in this workgroup.
    const uint x_offset = wgID.x * N0 + C_COLS * localID.x;
    const uint y_offset = wgID.y * M0 + C_ROWS * localID.y;
    
    int32_t C[C_ROWS][C_COLS]; // Local data for the output.
    
    // Initialize result to zero.
    for (uint i = 0; i < C_ROWS; ++i) {
        for (uint j = 0; j < C_COLS; ++j) {
            C[i][j] = 0;
        }
    }
    
    for (uint k = 0; k < K_VEC; k += K0_VEC) {
        for (uint i = 0; i < C_ROWS; ++i) {
            for (uint kk = 0; kk < K0_VEC; ++kk) {
                uint y = y_offset + i;
                uint gk = k + (kk + threadID) % K0_VEC;
                char4 lhs = inputA[coordToOffset(y, gk, strideA)];
                for (uint j = 0; j < C_COLS; ++j) {
                    // Calculate the inner product `C[i, j] := sum(A[i, ..] * B[j, ..])`.
                    uint x = x_offset + j;
                    char4 rhs = inputB[coordToOffset(x, gk, strideB)];
                    C[i][j] += sdot(lhs, rhs);
                }
            }
        }
    }
    
    // Store the accumulated results in `outputO`.
    for (uint i = 0; i < C_ROWS; ++i) {
        uint y = y_offset + i;
        for (uint j = 0; j < C_COLS; ++j) {
            uint x = x_offset + j;
            outputO[coordToOffset(y, x, strideC)] = C[i][j];
        }
    }
}



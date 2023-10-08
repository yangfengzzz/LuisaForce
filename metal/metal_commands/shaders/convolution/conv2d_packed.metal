//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <metal_stdlib>
using namespace metal;

uint3 wgID [[threadgroup_position_in_grid]];
uint3 threadID [[thread_position_in_threadgroup]];
uint3 threadCount [[threadgroups_per_grid]];

constant uint OH [[function_constant(0)]]; // Output height
constant uint OW [[function_constant(1)]]; // Output width
constant uint OC [[function_constant(2)]]; // Output channel
constant uint IH [[function_constant(3)]]; // Input height
constant uint IW [[function_constant(4)]]; // Input width
constant uint IC [[function_constant(5)]]; // Input channel
constant uint FH [[function_constant(6)]]; // Filter height
constant uint FW [[function_constant(7)]]; // Filter width
constant uint SH [[function_constant(8)]]; // Height stride
constant uint SW [[function_constant(9)]]; // Width stride

uint inputCoordToOffset(uint h, uint w, uint c) {
    return (h  * IW * IC + w * IC + c) / 4;
}

uint filterCoordToOffset(uint h, uint w, uint ic, uint oc) {
    return (h * FW * IC * OC + w  * IC * OC + ic * OC + oc) / 4;
}

uint outputCoordToOffset(uint h, uint w, uint c) {
    return (h  * OW * OC + w * OC + c) / 4;
}

#ifndef IVC_OH
#define IVC_OH 8
#endif

#ifndef IVC_OW
#define IVC_OW 8
#endif

#ifndef IVC_OC
#define IVC_OC 8
#endif

kernel void conv2d_packed(device float4* Input [[buffer(0)]],
                          device float4* Filter [[buffer(1)]],
                          device float4* Output [[buffer(2)]]) {
    // Each invocation calculates (IVC_OH * IVC_OW * IVC_OC * 8) output elements.
    half4 O[IVC_OH][IVC_OW][IVC_OC][2];
    
    // Use registers to keep the filter for this tile to increase data reuse.
    half4 F[8][IVC_OC][2];
    
    const uint WG_TILE_OH = threadCount.z * IVC_OH;
    const uint WG_TILE_OW = threadCount.y * IVC_OW;
    const uint WG_TILE_OC = threadCount.x * IVC_OC * 4;
    
    uint wgBaseOC = wgID.x * WG_TILE_OC; // Workgroup base output channel
    uint wgBaseOW = wgID.y * WG_TILE_OW; // Workgroup base output width
    uint wgBaseOH = wgID.z * WG_TILE_OH; // Workgroup base output height
    
    // Initialize the output for this batch to zero.
    for (uint i = 0; i < IVC_OH; ++i) {
        for (uint j = 0; j < IVC_OW; ++j) {
            for (uint k = 0; k < IVC_OC; ++k) {
                O[i][j][k][0] = half4(0.f, 0.f, 0.f, 0.f);
                O[i][j][k][1] = half4(0.f, 0.f, 0.f, 0.f);
            }
        }
    }
    
    for (uint fh = 0; fh < FH; ++fh) {
        for (uint fw = 0; fw < FW; ++fw) {
            // Tile input channel with each tile having 8 elements.
            for (uint ic = 0; ic < IC; ic += 8) {
                // Load the filter for this input channel tile.
                for (uint i = 0; i < 8; ++i) {
                    for (uint j = 0; j < IVC_OC; ++j) {
                        uint oc = (threadID.x + threadCount.x * j) * 8 + wgBaseOC;
                        float4 f_kernel = Filter[filterCoordToOffset(fh, fw, ic + i, oc)];
                        F[i][j][0].x = unpack_unorm2x16_to_half(as_type<uint>(f_kernel.x)).x;
                        F[i][j][0].y = unpack_unorm2x16_to_half(as_type<uint>(f_kernel.x)).y;
                        F[i][j][0].z = unpack_unorm2x16_to_half(as_type<uint>(f_kernel.y)).x;
                        F[i][j][0].w = unpack_unorm2x16_to_half(as_type<uint>(f_kernel.y)).y;
                        F[i][j][1].x = unpack_unorm2x16_to_half(as_type<uint>(f_kernel.z)).x;
                        F[i][j][1].y = unpack_unorm2x16_to_half(as_type<uint>(f_kernel.z)).y;
                        F[i][j][1].z = unpack_unorm2x16_to_half(as_type<uint>(f_kernel.w)).x;
                        F[i][j][1].w = unpack_unorm2x16_to_half(as_type<uint>(f_kernel.w)).y;
                    }
                }
                
                // Load this input channel tile and perform dot product with filters
                // for different output elements.
                for (uint i = 0; i < IVC_OH; ++i) {
                    uint oh = i + threadID.z * IVC_OH + wgBaseOH;
                    for (uint j = 0; j < IVC_OW; ++j) {
                        uint ow = j + threadID.y * IVC_OW + wgBaseOW;
                        float4 feature = Input[inputCoordToOffset(oh * SH + fh, ow * SW + fw, ic)];
                        for (uint k = 0; k < IVC_OC; ++k) {
                            for (uint l = 0; l < 2; ++l) {
                                half v;
                                v = unpack_unorm2x16_to_half(as_type<uint>(feature.x)).x;
                                O[i][j][k][l] += half4(v, v, v, v) * F[0][k][l];
                                v = unpack_unorm2x16_to_half(as_type<uint>(feature.x)).y;
                                O[i][j][k][l] += half4(v, v, v, v) * F[1][k][l];
                                v = unpack_unorm2x16_to_half(as_type<uint>(feature.y)).x;
                                O[i][j][k][l] += half4(v, v, v, v) * F[2][k][l];
                                v = unpack_unorm2x16_to_half(as_type<uint>(feature.y)).y;
                                O[i][j][k][l] += half4(v, v, v, v) * F[3][k][l];
                                v = unpack_unorm2x16_to_half(as_type<uint>(feature.z)).x;
                                O[i][j][k][l] += half4(v, v, v, v) * F[4][k][l];
                                v = unpack_unorm2x16_to_half(as_type<uint>(feature.z)).y;
                                O[i][j][k][l] += half4(v, v, v, v) * F[5][k][l];
                                v = unpack_unorm2x16_to_half(as_type<uint>(feature.w)).x;
                                O[i][j][k][l] += half4(v, v, v, v) * F[6][k][l];
                                v = unpack_unorm2x16_to_half(as_type<uint>(feature.w)).y;
                                O[i][j][k][l] += half4(v, v, v, v) * F[7][k][l];
                            }
                        }
                    }
                }
            }
        }
    }
    
    // Write out the computed output elements.
    for (uint i = 0; i < IVC_OH; ++i) {
        uint oh = i + threadID.z * IVC_OH + wgBaseOH;
        for (uint j = 0; j < IVC_OW; ++j) {
            uint ow = j + threadID.y * IVC_OW + wgBaseOW;
            for (uint k = 0; k < IVC_OC; ++k) {
                float4 result;
                result.x = as_type<float>(pack_half_to_unorm2x16(O[i][j][k][0].xy));
                result.y = as_type<float>(pack_half_to_unorm2x16(O[i][j][k][0].zw));
                result.z = as_type<float>(pack_half_to_unorm2x16(O[i][j][k][1].xy));
                result.w = as_type<float>(pack_half_to_unorm2x16(O[i][j][k][1].zw));
                uint oc = (threadID.x + threadCount.x * k) * 8 + wgBaseOC;
                Output[outputCoordToOffset(oh, ow, oc)] = result;
            }
        }
    }
}

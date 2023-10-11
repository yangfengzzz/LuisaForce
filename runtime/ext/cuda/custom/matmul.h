//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "cuda/cuda_buffer.h"
#include "cuda/math/vec.h"

#ifdef __CUDACC__

namespace luisa::compute::cuda {
__device__ inline uint coordToOffset(uint i, uint j, uint stride) {
    return (stride * i + j);
}

template<uint TILE_M, uint TILE_N, uint TILE_K, uint WG_X, uint WG_Y>
__global__ void matmul_tiled_fp32(CUdeviceptr src0, CUdeviceptr src1, CUdeviceptr dst, uint M, uint N, uint K) {
    auto *inputA = reinterpret_cast<vec4f *>(src0);
    auto *inputB = reinterpret_cast<vec4f *>(src1);
    auto *outputO = reinterpret_cast<vec4f *>(dst);

    ::uint2 gID{blockIdx.x, blockIdx.y};
    ::uint2 laneId{threadIdx.x, threadIdx.y};

    const uint strideA = K;
    const uint strideB = N;
    const uint strideC = N;

    const uint C_ROWS = TILE_M / WG_Y;
    const uint C_COLS = TILE_N / (4 * WG_X);

    vec4f C[C_ROWS][C_COLS];
    vec4f B[TILE_K][C_COLS];

    // Initialize result to zero
    for (uint i = 0; i < C_ROWS; ++i) {
        for (uint j = 0; j < C_COLS; ++j) {
            C[i][j] = vec4(0.f, 0.f, 0.f, 0.f);
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
            for (uint kk = 0; kk < TILE_K / 4; kk++) {
                vec4 A = inputA[coordToOffset(gi, gk + kk, strideA / 4)];
                for (uint j = 0; j < C_COLS; ++j) {
                    C[i][j] += cw_mul(vec4(A[0]), B[0 + 4 * kk][j]);
                    C[i][j] += cw_mul(vec4(A[1]), B[1 + 4 * kk][j]);
                    C[i][j] += cw_mul(vec4(A[2]), B[2 + 4 * kk][j]);
                    C[i][j] += cw_mul(vec4(A[3]), B[3 + 4 * kk][j]);
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

template<uint TILE_M, uint TILE_N, uint TILE_K, uint WG_X, uint WG_Y>
CudaCommand::UCommand CudaCommand::matmul(BufferView<float> src0_buffer, BufferView<float> src1_buffer, BufferView<float> dst_buffer,
                                          int M, int N, int K) noexcept {
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>(
        [=](CUstream stream) {
            dim3 gridDim(uint32_t(N / TILE_N), uint32_t(M / TILE_M));
            dim3 blockDim(WG_X, WG_Y);
            matmul_tiled_fp32<8, 8, 8, 1, 1><<<gridDim, blockDim, 0, stream>>>(
                reinterpret_cast<const CUDABuffer *>(src0_buffer.handle())->handle(),
                reinterpret_cast<const CUDABuffer *>(src1_buffer.handle())->handle(),
                reinterpret_cast<const CUDABuffer *>(dst_buffer.handle())->handle(),
                M, N, K);
        });
}

}// namespace luisa::compute::cuda

#endif
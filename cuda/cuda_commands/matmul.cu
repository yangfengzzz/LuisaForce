//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/ext/cuda/cuda_commands.h"
#include "cuda/cuda_buffer.h"
#include "cuda/math/vec.h"

namespace luisa::compute::cuda {
template<typename TYPE>
__global__ void matmul_tiled_fp32(CUdeviceptr src0, CUdeviceptr src1, CUdeviceptr dst, uint32_t kLoopSize) {
    auto *inputA = reinterpret_cast<TYPE *>(src0);
    auto *inputB = reinterpret_cast<TYPE *>(src1);
    auto *outputO = reinterpret_cast<TYPE *>(dst);

    auto index = grid_index();

    TYPE a = inputA[index];
    TYPE b = inputB[index];
    TYPE c = TYPE(1.f, 1.f, 1.f, 1.f);
    for (int i = 0; i < kLoopSize; i++) {
        c = cw_mul(a, c) + b;
        c = cw_mul(a, c) + b;
        c = cw_mul(a, c) + b;
        c = cw_mul(a, c) + b;
        c = cw_mul(a, c) + b;
        c = cw_mul(a, c) + b;
        c = cw_mul(a, c) + b;
        c = cw_mul(a, c) + b;
        c = cw_mul(a, c) + b;
        c = cw_mul(a, c) + b;
    }
    outputO[index] = c;
}

CudaCommand::UCommand CudaCommand::matmul(BufferView<float> src0_buffer, BufferView<float> src1_buffer, BufferView<float> dst_buffer,
                                          int tileM, int tileN, int tileK,
                                          int M, int N, int K,
                                          int wg_size_x, int wg_size_y) noexcept {
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>(
        [=](CUstream stream) {
            dim3 gridDim(uint32_t(N / tileN), uint32_t(M / tileM));
            dim3 blockDim(wg_size_x, wg_size_y);
            matmul_tiled_fp32<vec4f><<<gridDim, blockDim, 0, stream>>>(
                reinterpret_cast<const CUDABuffer *>(src0_buffer.handle())->handle(),
                reinterpret_cast<const CUDABuffer *>(src1_buffer.handle())->handle(),
                reinterpret_cast<const CUDABuffer *>(dst_buffer.handle())->handle(),
                src0_buffer.size());
        });
}

}// namespace luisa::compute::cuda
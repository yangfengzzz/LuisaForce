//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/ext/metal/metal_command.h"
#include "metal_buffer.h"

namespace luisa::compute::metal {
MetalCommand::UCommand MetalCommand::matmul(BufferView<float> src0_buffer, BufferView<float> src1_buffer, BufferView<float> dst_buffer,
                                            int tileM, int tileN, int tileK,
                                            int M, int N, int K,
                                            int wg_size_x, int wg_size_y) noexcept {
    return luisa::make_unique<luisa::compute::metal::MetalCommand>(
        [=](MTL::ComputeCommandEncoder *encoder, MTL::ComputePipelineState *pso) {
            encoder->setComputePipelineState(pso);
            encoder->setBuffer(reinterpret_cast<const MetalBuffer *>(src0_buffer.handle())->handle(), 0, 0);
            encoder->setBuffer(reinterpret_cast<const MetalBuffer *>(src1_buffer.handle())->handle(), 0, 1);
            encoder->setBuffer(reinterpret_cast<const MetalBuffer *>(dst_buffer.handle())->handle(), 0, 2);
            encoder->dispatchThreadgroups({uint32_t(N / tileN), uint32_t(M / tileM), 1}, {uint32_t(wg_size_x), uint32_t(wg_size_y), 1});
        },
        [=](MTL::Device *device) {
            std::string entry = "matmul_tiled_fp32";
            std::string shader_source = MetalCommand::read_shader("metal/metal_commands/shaders/matmul/matmul_tiled_fp32.metal");

            std::unordered_map<std::string, std::string> macros;
            macros["M"] = std::to_string(M);
            macros["N"] = std::to_string(N);
            macros["K"] = std::to_string(K);
            macros["TILE_M"] = std::to_string(tileM);
            macros["TILE_N"] = std::to_string(tileN);
            macros["TILE_K"] = std::to_string(tileK);
            macros["WG_X"] = std::to_string(wg_size_x);
            macros["WG_Y"] = std::to_string(wg_size_y);
            return create_pipeline_cache(device, shader_source, entry, macros);
        });
}

}// namespace luisa::compute::metal
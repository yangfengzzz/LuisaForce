//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/ext/metal/metal_command.h"
#include "metal_buffer.h"

namespace luisa::compute::metal {
MetalCommand::UCommand MetalCommand::one_workgroup_reduce(BufferView<float> src_buffer,
                                                          BufferView<float> dst_buffer,
                                                          size_t batch_elements) noexcept {
    return luisa::make_unique<luisa::compute::metal::MetalCommand>(
        [=](MTL::ComputeCommandEncoder *encoder, MTL::ComputePipelineState *pso) {
            encoder->setComputePipelineState(pso);
            encoder->setBuffer(reinterpret_cast<const MetalBuffer *>(src_buffer.handle())->handle(), 0, 0);
            encoder->setBuffer(reinterpret_cast<const MetalBuffer *>(dst_buffer.handle())->handle(), 0, 2);
//            encoder->dispatchThreads({(uint32_t)src0_buffer.size() / 4, 1, 1}, {pso->threadExecutionWidth(), 1, 1});
        },
        [=](MTL::Device *device) {
            std::string entry = "mad_throughput";
            std::string shader_source = MetalCommand::read_shader("metal/metal_commands/shaders/mad_throughput.metal");

            std::unordered_map<std::string, std::string> macros;
            return create_pipeline_cache(device, shader_source, entry, macros);
        });
}

}// namespace luisa::compute::metal
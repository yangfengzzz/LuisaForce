//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/ext/metal/metal_command.h"
#include "metal_buffer.h"

namespace luisa::compute::metal {
MetalCommand::UCommand MetalCommand::atomic_reduce(BufferView<float> src_buffer, BufferView<float> dst_buffer,
                                                   size_t batch_elements, bool is_integer) noexcept {
    return luisa::make_unique<luisa::compute::metal::MetalCommand>(
        [=](MTL::ComputeCommandEncoder *encoder, MTL::ComputePipelineState *pso) {
            encoder->setComputePipelineState(pso);
            encoder->setBuffer(reinterpret_cast<const MetalBuffer *>(src_buffer.handle())->handle(), 0, 0);
            encoder->setBuffer(reinterpret_cast<const MetalBuffer *>(dst_buffer.handle())->handle(), 0, 1);
            encoder->dispatchThreadgroups({uint32_t(src_buffer.size() / batch_elements), 1, 1}, {16, 1, 1});
        },
        [=](MTL::Device *device) {
            std::string entry;
            std::string shader_source;
            if (is_integer) {
                entry = "atomic_reduce_loop_int";
                shader_source = MetalCommand::read_shader("metal/metal_commands/shaders/matmul/atomic_reduce_loop_float.metal");
            } else {
                entry = "atomic_reduce_loop_float";
                shader_source = MetalCommand::read_shader("metal/metal_commands/shaders/matmul/atomic_reduce_loop_int.metal");
            }

            std::unordered_map<std::string, std::string> macros;
            macros["BATCH_SIZE"] = std::to_string(batch_elements);
            return create_pipeline_cache(device, shader_source, entry, macros);
        });
}

}// namespace luisa::compute::metal
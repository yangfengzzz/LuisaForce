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
                                                          size_t total_elements, ReduceMode mode) noexcept {
    return luisa::make_unique<luisa::compute::metal::MetalCommand>(
        [=](MTL::ComputeCommandEncoder *encoder, MTL::ComputePipelineState *pso) {
            encoder->setComputePipelineState(pso);
            encoder->setBuffer(reinterpret_cast<const MetalBuffer *>(src_buffer.handle())->handle(), 0, 0);
            encoder->setBuffer(reinterpret_cast<const MetalBuffer *>(dst_buffer.handle())->handle(), 0, 1);
            uint32_t thread_count = 16;
            if (mode == ReduceMode::Atomic) {
                thread_count = 1;
            }
            encoder->dispatchThreadgroups({1, 1, 1}, {thread_count, 1, 1});
        },
        [=](MTL::Device *device) {
            std::string entry;
            std::string shader_source;
            switch (mode) {
                case ReduceMode::Atomic:
                    entry = "one_workgroup_reduce_atomic";
                    shader_source = MetalCommand::read_shader("metal/metal_commands/shaders/reduction/one_workgroup_reduce_atomic.metal");
                    break;

                case ReduceMode::Loop:
                    entry = "one_workgroup_reduce_loop";
                    shader_source = MetalCommand::read_shader("metal/metal_commands/shaders/reduction/one_workgroup_reduce_loop.metal");
                    break;
                case ReduceMode::SimdGroup:
                    entry = "one_workgroup_reduce_subgroup";
                    shader_source = MetalCommand::read_shader("metal/metal_commands/shaders/reduction/one_workgroup_reduce_subgroup.metal");
                    break;
            }

            std::unordered_map<std::string, std::string> macros;
            macros["totalCount"] = std::to_string(total_elements);
            return create_pipeline_cache(device, shader_source, entry, macros);
        });
}

}// namespace luisa::compute::metal
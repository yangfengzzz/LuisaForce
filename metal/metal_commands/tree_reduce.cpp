//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/ext/metal/metal_command.h"
#include "metal_buffer.h"

namespace luisa::compute::metal {
MetalCommand::UCommand MetalCommand::tree_reduce(BufferView<float> buffer,
                                                 size_t batch_elements, ReduceMode mode, bool is_integer) noexcept {
    return luisa::make_unique<luisa::compute::metal::MetalCommand>(
        [=](MTL::ComputeCommandEncoder *encoder, MTL::ComputePipelineState *pso) {
            encoder->setComputePipelineState(pso);
            encoder->setBuffer(reinterpret_cast<const MetalBuffer *>(buffer.handle())->handle(), 0, 0);
            for (uint32_t batch = buffer.size() / batch_elements; batch > 0; batch /= batch_elements) {
                encoder->dispatchThreadgroups({uint32_t(batch), 1, 1}, {16, 1, 1});
            }
        },
        [=](MTL::Device *device) {
            std::string entry;
            std::string shader_source;
            switch (mode) {
                case ReduceMode::Loop:
                    entry = "tree_reduce_loop";
                    shader_source = MetalCommand::read_shader("metal/metal_commands/shaders/reduction/tree_reduce_loop.metal");
                    break;
                case ReduceMode::SimdGroup:
                    entry = "tree_reduce_subgroup";
                    shader_source = MetalCommand::read_shader("metal/metal_commands/shaders/reduction/tree_reduce_subgroup.metal");
                    break;
            }

            std::unordered_map<std::string, std::string> macros;
            macros["BATCH_SIZE"] = std::to_string(batch_elements);
            if (is_integer) {
                macros["TYPE"] = "int";
            } else {
                macros["TYPE"] = "float";
            }
            return create_pipeline_cache(device, shader_source, entry, macros);
        });
}

}// namespace luisa::compute::metal
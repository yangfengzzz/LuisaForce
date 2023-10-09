//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/ext/metal/metal_command.h"
#include "metal_buffer.h"

namespace luisa::compute::metal {
MetalCommand::UCommand MetalCommand::simd_group_arithmetic(BufferView<float> src_buffer, BufferView<float> dst_buffer,
                                                           size_t num_elements, ArithmeticMode mode) noexcept {
    static uint32_t kWorkgroupSize = 64;
    return luisa::make_unique<luisa::compute::metal::MetalCommand>(
        [=](MTL::ComputeCommandEncoder *encoder, MTL::ComputePipelineState *pso) {
            encoder->setComputePipelineState(pso);
            encoder->setBuffer(reinterpret_cast<const MetalBuffer *>(src_buffer.handle())->handle(), 0, 0);
            encoder->setBuffer(reinterpret_cast<const MetalBuffer *>(dst_buffer.handle())->handle(), 0, 1);
            encoder->dispatchThreadgroups({uint32_t(num_elements / kWorkgroupSize), 1, 1}, {kWorkgroupSize, 1, 1});
        },
        [=](MTL::Device *device) {
            std::string entry = "simdgroup_arithmetic_intrinsic";
            std::string shader_source = MetalCommand::read_shader("metal/metal_commands/shaders/simd_group/simdgroup_arithmetic_intrinsic.metal");

            std::unordered_map<std::string, std::string> macros;
            switch (mode) {
                case ArithmeticMode::Add:
                    macros["ARITHMETIC_ADD"] = std::to_string(true);
                    break;
                case ArithmeticMode::Mul:
                    macros["ARITHMETIC_MUL"] = std::to_string(true);
                    break;
            }
            macros["kArraySize"] = std::to_string(num_elements);

            return create_pipeline_cache(device, shader_source, entry, macros);
        });
}

}// namespace luisa::compute::metal
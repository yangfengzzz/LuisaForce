//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/ext/metal/metal_command.h"
#include "metal_buffer.h"

namespace luisa::compute::metal {
MetalCommand::UCommand MetalCommand::matmul(BufferView<float> src0_buffer, BufferView<float> src1_buffer, BufferView<float> dst_buffer,
                                            std::array<uint32_t, 3> threads, std::array<uint32_t, 3> thread_groups) noexcept {
    return luisa::make_unique<luisa::compute::metal::MetalCommand>(
        [=](MTL::ComputeCommandEncoder *encoder, MTL::ComputePipelineState *pso) {
            encoder->setComputePipelineState(pso);
            encoder->setBuffer(reinterpret_cast<const MetalBuffer *>(src0_buffer.handle())->handle(), 0, 0);
            encoder->setBuffer(reinterpret_cast<const MetalBuffer *>(src1_buffer.handle())->handle(), 0, 1);
            encoder->setBuffer(reinterpret_cast<const MetalBuffer *>(dst_buffer.handle())->handle(), 0, 2);
            encoder->dispatchThreads({threads[0], threads[1], threads[2]}, {thread_groups[0], thread_groups[1], thread_groups[2]});
        },
        [&](MTL::Device *device) {
            std::string entry = "matmul_tiled_fp32";
            std::string shader_source = "#include <metal_stdlib>\n"
                                        "using namespace metal;\n"
                                        "\n"
                                        "kernel void mad_throughput(device float4* inputA [[buffer(0)]],\n"
                                        "                           device float4* inputB [[buffer(1)]],\n"
                                        "                           device float4* output [[buffer(2)]],\n"
                                        "                           uint3 tpig [[ thread_position_in_grid ]]) {\n"
                                        "    float4 a = inputA[tpig.x];\n"
                                        "    float4 b = inputB[tpig.x];\n"
                                        "    float4 c = float4(1.f, 1.f, 1.f, 1.f);\n"
                                        "    for(int i = 0; i < kLoopSize; i++) {\n"
                                        "        c = a * c + b;\n"
                                        "        c = a * c + b;\n"
                                        "        c = a * c + b;\n"
                                        "        c = a * c + b;\n"
                                        "        c = a * c + b;\n"
                                        "        c = a * c + b;\n"
                                        "        c = a * c + b;\n"
                                        "        c = a * c + b;\n"
                                        "        c = a * c + b;\n"
                                        "        c = a * c + b;\n"
                                        "    }\n"
                                        "    output[tpig.x] = c;\n"
                                        "}";

            std::unordered_map<std::string, std::string> macros;
            macros["kLoopSize"] = std::to_string(src0_buffer.size());
            return create_pipeline_cache(device, shader_source, entry, macros);
        });
}

}// namespace luisa::compute::metal
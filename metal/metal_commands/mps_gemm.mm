//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/ext/metal/mps_command.h"
#include "metal_buffer.h"
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

namespace luisa::compute::metal {
MPSCommand::UCommand MPSCommand::gemm(BufferView<float> src0_buffer, BufferView<float> src1_buffer, BufferView<float> dst_buffer,
                                      int M, int N, int K) noexcept {
    return luisa::make_unique<luisa::compute::metal::MPSCommand>(
        [=](MTL::CommandBuffer *command_buffer, luisa::vector<NS::Object *> objects) {
            auto *mps = static_cast<MPSMatrixMultiplication *>(objects[0]);
            auto *left_m = static_cast<MPSMatrix *>(objects[1]);
            auto *right_m = static_cast<MPSMatrix *>(objects[2]);
            auto *result_m = static_cast<MPSMatrix *>(objects[3]);
            [mps encodeToCommandBuffer:(id<MTLCommandBuffer>)command_buffer
                            leftMatrix:left_m
                           rightMatrix:right_m
                          resultMatrix:result_m];
        },
        [=](MTL::Device *device) {
            auto left_desc = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                   columns:K
                                                                  rowBytes:M * sizeof(float)
                                                                  dataType:MPSDataTypeFloat32];
            auto right_desc = [MPSMatrixDescriptor matrixDescriptorWithRows:K
                                                                    columns:N
                                                                   rowBytes:K * sizeof(float)
                                                                   dataType:MPSDataTypeFloat32];
            auto result_desc = [MPSMatrixDescriptor matrixDescriptorWithRows:M
                                                                     columns:N
                                                                    rowBytes:M * sizeof(float)
                                                                    dataType:MPSDataTypeFloat32];
            auto left_m = (NS::Object *)[[MPSMatrix alloc] initWithBuffer:(id<MTLBuffer>)reinterpret_cast<const MetalBuffer *>(src0_buffer.handle())->handle()
                                                               descriptor:left_desc];
            auto right_m = (NS::Object *)[[MPSMatrix alloc] initWithBuffer:(id<MTLBuffer>)reinterpret_cast<const MetalBuffer *>(src1_buffer.handle())->handle()
                                                                descriptor:right_desc];
            auto result_m = (NS::Object *)[[MPSMatrix alloc] initWithBuffer:(id<MTLBuffer>)reinterpret_cast<const MetalBuffer *>(dst_buffer.handle())->handle()
                                                                 descriptor:result_desc];

            auto mps = (NS::Object *)[[MPSMatrixMultiplication alloc] initWithDevice:(id<MTLDevice>)device
                                                                          resultRows:M
                                                                       resultColumns:N
                                                                     interiorColumns:K];

            return luisa::vector<NS::Object *>{mps, left_m, right_m, result_m};
        });
}
}// namespace luisa::compute::metal
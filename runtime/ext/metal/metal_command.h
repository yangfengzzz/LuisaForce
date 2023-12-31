//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/stl/vector.h"
#include "core/stl/functional.h"
#include "runtime/buffer.h"
#include "runtime/device.h"
#include "runtime/stream.h"
#include "runtime/context.h"
#include "runtime/rhi/command.h"
#include "runtime/ext/registry.h"

namespace MTL {
class ComputeCommandEncoder;
class ComputePipelineState;
class Device;
}// namespace MTL

namespace luisa::compute::metal {

class MetalCommand final : public luisa::compute::CustomCommand {
public:
    template<typename T>
    using BufferView = luisa::compute::BufferView<T>;
    using UCommand = luisa::unique_ptr<luisa::compute::metal::MetalCommand>;

    luisa::function<void(MTL::ComputeCommandEncoder *encoder, MTL::ComputePipelineState *pso)> func;
    luisa::function<MTL::ComputePipelineState *(MTL::Device *device)> pso_func;

    MTL::ComputePipelineState *pso{nullptr};

    static MTL::ComputePipelineState *create_pipeline_cache(MTL::Device *device,
                                                            const std::string &raw_source, const std::string &entry,
                                                            const std::unordered_map<std::string, std::string> &macros);

    static std::string read_shader(const std::string &filename);

public:
    MetalCommand(luisa::function<void(MTL::ComputeCommandEncoder *encoder, MTL::ComputePipelineState *pso)> f,
                 luisa::function<MTL::ComputePipelineState *(MTL::Device *device)> pso_f) noexcept
        : CustomCommand{}, func{std::move(f)}, pso_func{std::move(pso_f)} {}

    explicit MetalCommand(MetalCommand *command) noexcept;

    ~MetalCommand() override;

    [[nodiscard]] StreamTag stream_tag() const noexcept override { return StreamTag::COMPUTE; }

    [[nodiscard]] uint64_t uuid() const noexcept override {
        return static_cast<uint64_t>(CustomCommandUUID::CUSTOM_DISPATCH);
    }

    void alloc_pso(Device *device);

    UCommand clone();

public:
    static UCommand mad_throughput(BufferView<float> src0_buffer, BufferView<float> src1_buffer, BufferView<float> dst_buffer) noexcept;

    static UCommand matmul(BufferView<float> src0_buffer, BufferView<float> src1_buffer, BufferView<float> dst_buffer,
                           int tileM, int tileN, int tileK,
                           int M, int N, int K,
                           int wg_size_x, int wg_size_y) noexcept;

    // matrix-matrix transposed multiplication of two 2D inputs.
    static UCommand mmt(BufferView<float> src0_buffer, BufferView<float> src1_buffer, BufferView<float> dst_buffer,
                        int tileM, int tileN, int tileK,
                        int M, int N, int K,
                        int wg_size_x, int wg_size_y) noexcept;

    enum class ReduceMode {
        Loop,
        SimdGroup
    };

    static UCommand atomic_reduce(BufferView<float> src_buffer, BufferView<float> dst_buffer,
                                  size_t batch_elements, ReduceMode mode, bool is_integer) noexcept;

    static UCommand one_workgroup_reduce(BufferView<float> src_buffer, BufferView<float> dst_buffer,
                                         size_t batch_elements, ReduceMode mode) noexcept;

    static UCommand tree_reduce(BufferView<float> buffer,
                                size_t batch_elements, ReduceMode mode, bool is_integer) noexcept;

    enum class ArithmeticMode {
        Add,
        Mul
    };

    static UCommand simd_group_arithmetic(BufferView<float> src_buffer, BufferView<float> dst_buffer,
                                          size_t batch_elements, ArithmeticMode mode) noexcept;
};

}// namespace luisa::compute::metal

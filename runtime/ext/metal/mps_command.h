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
class CommandBuffer;
class Device;
}// namespace MTL

namespace NS {
class Object;
}// namespace NS

namespace luisa::compute::metal {
// metal performance shader
class MPSCommand final : public luisa::compute::CustomCommand {
public:
    template<typename T>
    using BufferView = luisa::compute::BufferView<T>;
    using UCommand = luisa::unique_ptr<luisa::compute::metal::MPSCommand>;

    luisa::function<void(MTL::CommandBuffer *cb, luisa::vector<NS::Object *> objects)> func;
    luisa::function<luisa::vector<NS::Object *>(MTL::Device *device)> kernel_func;

    luisa::vector<NS::Object *> objects{};

public:
    MPSCommand(luisa::function<void(MTL::CommandBuffer *cb, luisa::vector<NS::Object *> objects)> f,
               luisa::function<luisa::vector<NS::Object *>(MTL::Device *device)> kernel_func) noexcept
        : CustomCommand{}, func{std::move(f)}, kernel_func{std::move(kernel_func)} {}

    explicit MPSCommand(MPSCommand *command) noexcept;

    ~MPSCommand() override;

    [[nodiscard]] StreamTag stream_tag() const noexcept override { return StreamTag::COMPUTE; }

    [[nodiscard]] uint64_t uuid() const noexcept override {
        return static_cast<uint64_t>(CustomCommandUUID::PERFORMANCE_DISPATCH);
    }

    UCommand clone();

public:
    static UCommand gemm(BufferView<float> src0_buffer, BufferView<float> src1_buffer, BufferView<float> dst_buffer,
                         int M, int N, int K) noexcept;
};

}// namespace luisa::compute::metal

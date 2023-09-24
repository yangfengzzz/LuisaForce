//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <Metal/Metal.hpp>
#include "core/stl/functional.h"
#include "runtime/buffer.h"
#include "runtime/device.h"
#include "runtime/stream.h"
#include "runtime/context.h"
#include "runtime/rhi/command.h"
#include "runtime/ext/registry.h"

namespace luisa::compute::metal {

class MetalCommand final : public luisa::compute::CustomCommand {
public:
    luisa::function<void(MTL::CommandEncoder *encoder, uint32_t width)> func;

    std::string shader_source;

    std::unordered_map<std::string, std::string> macros;

public:
    explicit MetalCommand(luisa::function<void(MTL::CommandEncoder *encoder, uint32_t thread_execution_width)> f,
                          std::string shader_source,
                          std::unordered_map<std::string, std::string> macros) noexcept
        : CustomCommand{}, func{std::move(f)},
          shader_source{std::move(shader_source)},
          macros{std::move(macros)} {}

    [[nodiscard]] StreamTag stream_tag() const noexcept override { return StreamTag::COMPUTE; }

    [[nodiscard]] uint64_t uuid() const noexcept override {
        return static_cast<uint64_t>(CustomCommandUUID::CUSTOM_DISPATCH);
    }
};

}// namespace luisa::compute::metal

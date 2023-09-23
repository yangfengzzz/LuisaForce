//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <cuda.h>
#include "core/stl/functional.h"
#include "runtime/buffer.h"
#include "runtime/device.h"
#include "runtime/stream.h"
#include "runtime/context.h"
#include "runtime/rhi/command.h"
#include "runtime/ext/registry.h"

namespace luisa::compute::cuda {

class CudaLCubCommand final : public luisa::compute::CustomCommand {
public:
    luisa::function<void(CUstream)> func;

public:
    explicit CudaLCubCommand(luisa::function<void(CUstream)> f) noexcept
        : CustomCommand{}, func{std::move(f)} {}
    [[nodiscard]] StreamTag stream_tag() const noexcept override { return StreamTag::COMPUTE; }
    [[nodiscard]] uint64_t uuid() const noexcept override {
        return static_cast<uint64_t>(CustomCommandUUID::CUDA_LCUB_COMMAND);
    }
};

}// namespace luisa::compute::cuda

//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <atomic>
#include "runtime/rhi/command.h"
#include "runtime/rhi/resource.h"
#include "runtime/rhi/stream_tag.h"
#include "runtime/buffer.h"

namespace luisa::compute {

class Device;

class LC_RUNTIME_API HashGrid final : public Resource {
private:
    friend class Device;
    HashGrid(DeviceInterface *device, int dim_x, int dim_y, int dim_z) noexcept;

public:
    HashGrid() noexcept = default;
    ~HashGrid() noexcept override;
    using Resource::operator bool;
    HashGrid(HashGrid &&) noexcept;
    HashGrid(HashGrid const &) noexcept = delete;
    HashGrid &operator=(HashGrid &&rhs) noexcept {
        _move_from(std::move(rhs));
        return *this;
    }
    HashGrid &operator=(HashGrid const &) noexcept = delete;

    auto build(BufferView<float3> source, float radius) noexcept;

    auto reserve(int num_points) noexcept;
};

}// namespace luisa::compute
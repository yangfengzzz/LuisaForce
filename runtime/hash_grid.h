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

namespace luisa::compute {

class Device;

class LC_RUNTIME_API HashGrid final : public Resource {
public:
    HashGrid(DeviceInterface *device, int dim_x, int dim_y, int dim_z) noexcept;
};

}// namespace luisa::compute
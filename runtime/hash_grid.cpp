//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/hash_grid.h"
#include "runtime/device.h"
#include "runtime/shader.h"

namespace luisa::compute {
namespace detail {

ShaderInvokeBase &ShaderInvokeBase::operator<<(const HashGrid &accel) noexcept {
    accel._check_is_valid();
    _encoder.encode_hash_grid(accel.handle());
    return *this;
}

}// namespace detail

// counting event
HashGrid Device::create_hash_grid(int dim_x, int dim_y, int dim_z) noexcept {
    return _create<HashGrid>(dim_x, dim_y, dim_z);
}

HashGrid::HashGrid(DeviceInterface *device, int dim_x, int dim_y, int dim_z) noexcept
    : Resource{device, Tag::HASH_GRID, device->create_hash_grid(dim_x, dim_y, dim_z)} {
}

HashGrid::HashGrid(HashGrid &&rhs) noexcept
    : Resource{std::move(rhs)} {}

HashGrid::~HashGrid() noexcept {
    if (*this) { device()->destroy_hash_grid(handle()); }
}

}// namespace luisa::compute

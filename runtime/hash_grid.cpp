//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/hash_grid.h"
#include "runtime/device.h"

namespace luisa::compute {
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

void HashGrid::build(float radius) {
    // todo
}

void HashGrid::reserve(int num_points) {
    // todo
}

}// namespace luisa::compute

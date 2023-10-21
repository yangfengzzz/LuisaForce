//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "cuda_builtin/math/cuda_vec.h"

namespace luisa::compute::cuda {
uint64_t hash_grid_create_device(void *context, int dim_x, int dim_y, int dim_z, CUstream stream);
void hash_grid_reserve_device(uint64_t id, int num_points, CUstream stream);
void hash_grid_destroy_device(uint64_t id);
void hash_grid_update_device(uint64_t id, float cell_width, const wp::vec3 *positions, int num_points, CUstream stream);
}// namespace luisa::compute::cuda

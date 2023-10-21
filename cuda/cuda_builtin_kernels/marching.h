//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "cuda_builtin/math/cuda_vec.h"

namespace luisa::compute::cuda {
uint64_t marching_cubes_create_device(void *context);
void marching_cubes_destroy_device(uint64_t id);
int marching_cubes_surface_device(uint64_t id, const float *field, int nx, int ny, int nz, float threshold,
                                  wp::vec3 *verts, int *triangles, int max_verts, int max_tris, int *out_num_verts, int *out_num_tris);
}// namespace luisa::compute::cuda

//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "cuda_util.h"
#include "cuda_context.h"
#include "hashgrid.h"
#include "sort.h"
#include "cuda_builtin/stl/hashgrid.h"

#include <cstring>
#include <map>

namespace luisa::compute::cuda {
namespace {
// host-side copy of mesh descriptors, maps GPU mesh address (id) to a CPU desc
std::map<uint64_t, wp::HashGrid> g_hash_grid_descriptors;

}// anonymous namespace

bool hash_grid_get_descriptor(uint64_t id, wp::HashGrid &grid) {
    const auto &iter = g_hash_grid_descriptors.find(id);
    if (iter == g_hash_grid_descriptors.end())
        return false;
    else
        grid = iter->second;
    return true;
}

void hash_grid_add_descriptor(uint64_t id, const wp::HashGrid &grid) {
    g_hash_grid_descriptors[id] = grid;
}

void hash_grid_rem_descriptor(uint64_t id) {
    g_hash_grid_descriptors.erase(id);
}

// implemented in hashgrid.cu
void hash_grid_rebuild_device(const wp::HashGrid &grid, const wp::vec3 *points, int num_points, CUstream stream);

// device methods
uint64_t hash_grid_create_device(void *context, int dim_x, int dim_y, int dim_z) {
    ContextGuard guard(context);

    wp::HashGrid grid;
    memset(&grid, 0, sizeof(wp::HashGrid));

    grid.context = context ? context : cuda_context_get_current();

    grid.dim_x = dim_x;
    grid.dim_y = dim_y;
    grid.dim_z = dim_z;

    const int num_cells = dim_x * dim_y * dim_z;
    grid.cell_starts = (int *)alloc_device(WP_CURRENT_CONTEXT, num_cells * sizeof(int));
    grid.cell_ends = (int *)alloc_device(WP_CURRENT_CONTEXT, num_cells * sizeof(int));

    // upload to device
    auto *grid_device = (wp::HashGrid *)(alloc_device(WP_CURRENT_CONTEXT, sizeof(wp::HashGrid)));
    memcpy_h2d(WP_CURRENT_CONTEXT, grid_device, &grid, sizeof(wp::HashGrid), nullptr);

    auto grid_id = (uint64_t)(grid_device);
    hash_grid_add_descriptor(grid_id, grid);

    return grid_id;
}

void hash_grid_destroy_device(uint64_t id) {
    wp::HashGrid grid;
    if (hash_grid_get_descriptor(id, grid)) {
        ContextGuard guard(grid.context);

        free_device(WP_CURRENT_CONTEXT, grid.point_ids);
        free_device(WP_CURRENT_CONTEXT, grid.point_cells);
        free_device(WP_CURRENT_CONTEXT, grid.cell_starts);
        free_device(WP_CURRENT_CONTEXT, grid.cell_ends);

        free_device(WP_CURRENT_CONTEXT, (wp::HashGrid *)id);

        hash_grid_rem_descriptor(id);
    }
}

void hash_grid_reserve_device(uint64_t id, int num_points, CUstream stream) {
    wp::HashGrid grid;

    if (hash_grid_get_descriptor(id, grid)) {
        if (num_points > grid.max_points) {
            ContextGuard guard(grid.context);

            free_device(WP_CURRENT_CONTEXT, grid.point_cells);
            free_device(WP_CURRENT_CONTEXT, grid.point_ids);

            const int num_to_alloc = num_points * 3 / 2;
            grid.point_cells = (int *)alloc_device(WP_CURRENT_CONTEXT, 2 * num_to_alloc * sizeof(int));// *2 for auxilliary radix buffers
            grid.point_ids = (int *)alloc_device(WP_CURRENT_CONTEXT, 2 * num_to_alloc * sizeof(int));  // *2 for auxilliary radix buffers
            grid.max_points = num_to_alloc;

            // ensure we pre-size our sort routine to avoid
            // allocations during graph capture
            radix_sort_reserve(WP_CURRENT_CONTEXT, num_to_alloc, nullptr, nullptr, stream);

            // update device side grid descriptor, todo: this is
            // slightly redundant since it is performed again
            // inside hash_grid_update_device(), but since
            // reserve can be called from Python we need to make
            // sure it is consistent
            memcpy_h2d(WP_CURRENT_CONTEXT, (wp::HashGrid *)id, &grid, sizeof(wp::HashGrid), stream);

            // update host side grid descriptor
            hash_grid_add_descriptor(id, grid);
        }
    }
}

void hash_grid_update_device(uint64_t id, float cell_width, const wp::vec3 *points, int num_points, CUstream stream) {

    // ensure we have enough memory reserved for update
    // this must be done before retrieving the descriptor
    // below since it may update it
    hash_grid_reserve_device(id, num_points, stream);

    // host grid must be static so that we can
    // perform host->device memcpy from this variable
    // and have it safely recorded inside CUDA graphs
    static wp::HashGrid grid;

    if (hash_grid_get_descriptor(id, grid)) {
        ContextGuard guard(grid.context);

        grid.num_points = num_points;
        grid.cell_width = cell_width;
        grid.cell_width_inv = 1.0f / cell_width;

        hash_grid_rebuild_device(grid, points, num_points, stream);

        // update device side grid descriptor
        memcpy_h2d(WP_CURRENT_CONTEXT, (wp::HashGrid *)id, &grid, sizeof(wp::HashGrid), stream);

        // update host side grid descriptor
        hash_grid_add_descriptor(id, grid);
    }
}

}// namespace luisa::compute::cuda
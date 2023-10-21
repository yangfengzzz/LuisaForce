//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "cuda_context.h"
#include "cuda_util.h"
#include "hashgrid.h"
#include "sort.h"

namespace luisa::compute::cuda {

__global__ void compute_cell_indices(HashGrid grid, const vec3 *points, int num_points) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_points) {
        grid.point_cells[tid] = hash_grid_index(grid, points[tid]);
        grid.point_ids[tid] = tid;
    }
}

__global__ void compute_cell_offsets(int *cell_starts, int *cell_ends, const int *point_cells, int num_points) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // compute cell start / end
    if (tid < num_points) {
        // scan the particle-cell array to find the start and end
        const int c = point_cells[tid];

        if (tid == 0)
            cell_starts[c] = 0;
        else {
            const int p = point_cells[tid - 1];

            if (c != p) {
                cell_starts[c] = tid;
                cell_ends[p] = tid;
            }
        }

        if (tid == num_points - 1) {
            cell_ends[c] = tid + 1;
        }
    }
}

void hash_grid_rebuild_device(const HashGrid &grid, const vec3 *points, int num_points) {
    ContextGuard guard(grid.context);

    wp_launch_device(WP_CURRENT_CONTEXT, compute_cell_indices, num_points, (grid, points, num_points));

    radix_sort_pairs_device(WP_CURRENT_CONTEXT, grid.point_cells, grid.point_ids, num_points);

    const int num_cells = grid.dim_x * grid.dim_y * grid.dim_z;

    memset_device(WP_CURRENT_CONTEXT, grid.cell_starts, 0, sizeof(int) * num_cells);
    memset_device(WP_CURRENT_CONTEXT, grid.cell_ends, 0, sizeof(int) * num_cells);

    wp_launch_device(WP_CURRENT_CONTEXT, compute_cell_offsets, num_points, (grid.cell_starts, grid.cell_ends, grid.point_cells, num_points));
}

}// namespace luisa::compute::cuda
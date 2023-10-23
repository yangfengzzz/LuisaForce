//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/context.h"
#include "runtime/stream.h"
#include "runtime/image.h"
#include "runtime/shader.h"
#include <runtime/hash_grid.h>
#include "dsl/syntax.h"

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context context{argv[0]};
    Device device = context.create_device();
    Stream stream = device.create_stream();

    // NOTE change this to adjust number of particles
    auto smoothing_length = 0.8;
    // x
    auto width = 80.0;
    // y
    auto height = 80.0;
    // number particles (small box in corner)
    auto n = int(height * (width / 4.0) * (height / 4.0) / (pow(smoothing_length, 3)));
    auto grid_size = int(height / (4.0 * smoothing_length));

    HashGrid hash_grid = device.create_hash_grid(grid_size, grid_size, grid_size);
    Buffer<float3> particle_x = device.create_buffer<float3>(n);

    constexpr uint2 resolution = make_uint2(1024, 1024);
    Kernel2D kernel = [&]() {
        Var coord = dispatch_id().x;
        // order threads by cell
        auto i = hash_grid->point_id(coord);

        // get local particle variables
        auto x = particle_x->read(i);

        // particle contact
        hash_grid->query(x, 2.0);
    };
    Shader2D<> shader = device.compile(kernel);
    stream << shader().dispatch(resolution)
           << synchronize();
}

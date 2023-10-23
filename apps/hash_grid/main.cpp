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
    HashGrid hash_grid = device.create_hash_grid(32, 32, 32);
    Buffer<float3> u1 = device.create_buffer<float3>(100);

    constexpr uint2 resolution = make_uint2(1024, 1024);
    Kernel2D kernel = [&]() {
        Var coord = dispatch_id().xy();
        auto i = hash_grid->point_id(coord);
    };
    Shader2D<> shader = device.compile(kernel);
    stream << shader().dispatch(resolution)
           << synchronize();
}

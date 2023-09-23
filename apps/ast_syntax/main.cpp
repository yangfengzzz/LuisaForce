//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/logging.h"
#include "runtime/context.h"
#include "runtime/device.h"
#include "runtime/stream.h"
#include "dsl/syntax.h"

using namespace luisa;
using namespace luisa::compute;
using luisa::compute::detail::FunctionBuilder;

int main(int argc, char *argv[]) {
    luisa::log_level_verbose();
    Context context{argv[0]};

    Device device = context.create_device();
    Stream stream = device.create_stream();

    LUISA_INFO("Buffer<int> description: {}", Type::of<Buffer<int>>()->description());

    Buffer<int> buf = device.create_buffer<int>(100);

    auto h = 1.0_h;
    auto f = sin(h);

    LUISA_INFO("h = {}, f = {}, f * f = {}, f + h = {}", h, f, f * f, f + h);

    Kernel1D k1 = [&] {
        buf->write(1, 42);
    };
    Shader1D<> s = device.compile(k1);
    stream << s().dispatch(1u);
    stream << synchronize();
}

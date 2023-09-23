//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/logging.h"
#include "core/clock.h"
#include "runtime/context.h"
#include "runtime/device.h"
#include "runtime/stream.h"
#include "runtime/event.h"
#include "runtime/swapchain.h"
#include "dsl/syntax.h"
#include "dsl/sugar.h"
#include "gui/window.h"

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {

    Context context{argv[0]};
    Device device = context.create_device();

    constexpr uint2 resolution = make_uint2(1280, 720);

    Stream stream = device.create_stream(StreamTag::GRAPHICS);
    Image<float> device_image1 = device.create_image<float>(PixelStorage::BYTE4, resolution);
    BindlessArray bdls = device.create_bindless_array();
    Buffer<float4> buffer = device.create_buffer<float4>(4);
    std::vector<float4> a{4};
    a[0] = {1, 0, 0, 1};
    a[1] = {0, 1, 0, 1};
    a[2] = {0, 0, 1, 1};
    a[3] = {1, 1, 1, 1};
    stream << buffer.copy_from(a.data()) << synchronize();
    bdls.emplace_on_update(0, buffer);
    stream << bdls.update() << synchronize();

    Kernel2D kernel = [&](Float time) {
        Var coord = dispatch_id().xy();
        UInt i2 = ((coord.x + cast<uint>(time)) / 16 % 4);
        auto vertex_array = bdls->buffer<float4>(0);
        Float4 p = vertex_array.read(i2);
        device_image1->write(coord, make_float4(p));
    };
    Shader2D<float> s = device.compile(kernel);

    Window window{"Display", resolution};

    Swapchain swapchain = device.create_swapchain(
        window.native_handle(), stream, resolution, false);
    Clock clk;
    while (!window.should_close()) {
        stream << s(static_cast<float>(clk.toc() * .05f))
                      .dispatch(1280, 720)
               << swapchain.present(device_image1);
        window.poll_events();
    }
}

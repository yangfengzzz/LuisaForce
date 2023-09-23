//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/logging.h"
#include "runtime/context.h"
#include "runtime/device.h"
#include "runtime/stream.h"
#include "runtime/event.h"
#include "dsl/syntax.h"

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    log_level_verbose();

    Context context{argv[0]};
    Device device = context.create_device();
    BindlessArray heap = device.create_bindless_array(4);
    Stream stream = device.create_stream();
    Buffer<int> buffer0 = device.create_buffer<int>(1);
    Buffer<int> buffer1 = device.create_buffer<int>(1);
    Buffer<int> out_buffer = device.create_buffer<int>(2);
    heap.emplace_on_update(0, buffer0);
    heap.emplace_on_update(1, buffer1);
    Kernel1D kernel = [&] {
        out_buffer->write(dispatch_id().x, heap->buffer<int>(dispatch_id().x).read(0));
    };
    Shader1D<> shader = device.compile(kernel);
    int v0 = 555;
    int v1 = 666;
    int result[2];
    stream << heap.update() << synchronize();
    stream << buffer0.copy_from(&v0) << buffer1.copy_from(&v1) << shader().dispatch(2) << out_buffer.copy_to(result) << synchronize();
    LUISA_INFO("Value: {}, {}", result[0], result[1]);
}

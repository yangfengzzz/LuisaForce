#include "runtime/context.h"
#include "runtime/stream.h"
#include "runtime/image.h"
#include "runtime/shader.h"
#include "dsl/syntax.h"
#include <stb_image_write.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context context{argv[0]};
    Device device = context.create_device();
    Stream stream = device.create_stream();
    constexpr uint2 resolution = make_uint2(1024, 1024);
    Image<float> image{device.create_image<float>(PixelStorage::BYTE4, resolution)};
    luisa::vector<std::byte> host_image(image.view().size_bytes());
    ExternalCallable<float2(float2, float2)> get_uv{"get_uv"};
    Kernel2D kernel = [&]() {
        Var coord = dispatch_id().xy();
        Var size = dispatch_size().xy();
        Var uv = get_uv(make_float2(coord), make_float2(size));
        image->write(coord, make_float4(uv, 0.5f, 1.0f));
    };
    ShaderOption option;
#if defined(LUISA_PLATFORM_CUDA)
    // native CUDA code
    option.native_include = R"(
[[nodiscard]] __device__ inline auto get_uv(lc_float2 coord, lc_float2 size) noexcept {
    return (coord + .5f) / size;
}
    )";
#endif

#if defined(LUISA_PLATFORM_APPLE)
    option.native_include = R"(
[[nodiscard]] inline auto get_uv(float2 coord, float2 size) {
    return (coord + .5f) / size;
}
    )";
#endif

    Shader2D<> shader = device.compile(kernel, option);
    stream << shader().dispatch(resolution)
           << image.copy_to(host_image.data())
           << synchronize();
    stbi_write_png("test_native_code.png", resolution.x, resolution.y, 4, host_image.data(), 0);
}

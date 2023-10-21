//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <runtime/context.h>
#include <runtime/stream.h>
#include <runtime/image.h>
#include <runtime/shader.h>
#include <runtime/hash_grid.h>
#include <dsl/sugar.h>
#include <dsl/syntax.h>
#include <runtime/swapchain.h>
#include <gui/window.h>

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char *argv[]) {
    Context context{argv[0]};
    DeviceConfig config{
        .device_index = 1};
    Device device = context.create_device(&config);
    Stream stream = device.create_stream(StreamTag::GRAPHICS);

    HashGrid hash_grid = device.create_hash_grid(32, 32, 32);

    static constexpr int n_grid = 1024;// care type
    Window window{"Fluid", n_grid, n_grid};
    Swapchain swap_chain{device.create_swapchain(
        window.native_handle(),
        stream,
        make_uint2(n_grid, n_grid),
        false, false, 2)};
    Image<float> display = device.create_image<float>(swap_chain.backend_storage(), make_uint2(n_grid, n_grid));

    auto index = [](const Int2 &xy) noexcept {
        auto p = clamp(xy, static_cast<int2>(0), static_cast<int2>(n_grid - 1));
        return p.x + p.y * n_grid;
    };

    Callable lookup_float = [&](BufferVar<float> f, const Int &x, const Int &y) noexcept {
        return f.read(index(make_int2(x, y)));
    };

    Callable sample_float = [&](BufferVar<float> f, const Float &x, const Float &y) noexcept {
        auto lx = cast<int>(floor(x));
        auto ly = cast<int>(floor(y));

        auto tx = x - cast<float>(lx);
        auto ty = y - cast<float>(ly);

        auto s0 = lerp(lookup_float(f, lx, ly), lookup_float(f, lx + 1, ly), tx);
        auto s1 = lerp(lookup_float(f, lx, ly + 1), lookup_float(f, lx + 1, ly + 1), tx);

        return lerp(s0, s1, ty);
    };

    Callable lookup_vel = [&](BufferVar<float2> f, const Int &x, const Int &y) noexcept {
        return f.read(index(make_int2(x, y)));
    };

    Callable sample_vel = [&](BufferVar<float2> f, const Float &x, const Float &y) noexcept {
        auto lx = cast<int>(floor(x));
        auto ly = cast<int>(floor(y));

        auto tx = x - cast<float>(lx);
        auto ty = y - cast<float>(ly);

        auto s0 = lerp(lookup_vel(f, lx, ly), lookup_vel(f, lx + 1, ly), tx);
        auto s1 = lerp(lookup_vel(f, lx, ly + 1), lookup_vel(f, lx + 1, ly + 1), tx);
        return lerp(s0, s1, ty);
    };

    auto advect = device.compile<2>([&](BufferVar<float2> u0, BufferVar<float2> u1,
                                        BufferVar<float> rho0, BufferVar<float> rho1, const Float &dt) noexcept {
        UInt2 coord = dispatch_id().xy();
        auto u = u0.read(index(coord));

        // trace backward
        auto p = Float2(cast<float>(coord.x), cast<float>(coord.y));
        p = p - u * dt;

        // advect
        u1.write(index(coord), sample_vel(u0, p[0], p[1]));
        rho1.write(index(coord), sample_float(rho0, p[0], p[1]));
    });

    auto divergence = device.compile<2>([&](BufferVar<float2> u, BufferVar<float> div) noexcept {
        UInt2 coord = dispatch_id().xy();

        $if((coord.x < n_grid - 1) & (coord.y < n_grid - 1)) {
            auto dx = (u.read(index(make_uint2(coord.x + 1, coord.y)))[0] - u.read(index(coord))[0]) * 0.5f;
            auto dy = (u.read(index(make_uint2(coord.x, coord.y + 1)))[1] - u.read(index(coord))[1]) * 0.5f;

            div.write(index(coord), dx + dy);
        };
    });

    auto pressure_solve = device.compile<2>([&](BufferVar<float> p0, BufferVar<float> p1, BufferVar<float> div) noexcept {
        UInt2 coord = dispatch_id().xy();
        auto i = cast<int>(coord.x);
        auto j = cast<int>(coord.y);
        auto ij = index(coord);

        auto s1 = lookup_float(p0, i - 1, j);
        auto s2 = lookup_float(p0, i + 1, j);
        auto s3 = lookup_float(p0, i, j - 1);
        auto s4 = lookup_float(p0, i, j + 1);

        // Jacobi update
        auto err = s1 + s2 + s3 + s4 - div.read(ij);
        p1.write(ij, err * 0.25f);
    });

    auto pressure_apply = device.compile<2>([&](BufferVar<float> p, BufferVar<float2> u) noexcept {
        UInt2 coord = dispatch_id().xy();
        auto i = cast<int>(coord.x);
        auto j = cast<int>(coord.y);
        auto ij = index(coord);

        $if((i > 0) & (i < n_grid - 1) & (j > 0) & (j < n_grid - 1)) {
            // pressure gradient
            auto f_p = Float2(p.read(index(make_int2(i + 1, j))) - p.read(index(make_int2(i - 1, j))),
                              p.read(index(make_int2(i, j + 1))) - p.read(index(make_int2(i, j - 1)))) *
                       0.5f;

            u.write(ij, u.read(ij) - f_p);
        };
    });

    auto integrate = device.compile<2>([&](BufferVar<float2> u, BufferVar<float> rho, const Float &dt) noexcept {
        UInt2 coord = dispatch_id().xy();
        auto ij = index(coord);

        // gravity
        auto f_g = Float2(-90.8f, 0.0f) * rho.read(ij);

        // integrate
        u.write(ij, u.read(ij) + dt * f_g);

        // fade
        rho.write(ij, rho.read(ij) * (1.0f - 0.1f * dt));
    });

    auto init = device.compile<2>([&](BufferVar<float> rho, BufferVar<float2> u, const Float &radius, Float2 dir) noexcept {
        UInt2 coord = dispatch_id().xy();
        auto i = cast<int>(coord.x);
        auto j = cast<int>(coord.y);
        auto ij = index(coord);

        auto d = length(make_float2(cast<float>(i - n_grid / 2), cast<float>(j - n_grid / 2)));
        $if(d < radius) {
            rho.write(ij, 1.0f);
            u.write(ij, dir);
        };
    });

    auto sim_fps = 60.0f;
    uint sim_substeps = 2;
    uint iterations = 100;
    auto dt = (1.0f / sim_fps) / static_cast<float>(sim_substeps);
    auto sim_time = 0.0f;
    auto speed = 400.0f;

    Buffer<float2> u0 = device.create_buffer<float2>(n_grid * n_grid);
    Buffer<float2> u1 = device.create_buffer<float2>(n_grid * n_grid);

    Buffer<float> rho0 = device.create_buffer<float>(n_grid * n_grid);
    Buffer<float> rho1 = device.create_buffer<float>(n_grid * n_grid);

    Buffer<float> p0 = device.create_buffer<float>(n_grid * n_grid);
    Buffer<float> p1 = device.create_buffer<float>(n_grid * n_grid);
    Buffer<float> div = device.create_buffer<float>(n_grid * n_grid);

    Shader2D<> init_grid = device.compile<2>([&] {
        UInt idx = index(dispatch_id().xy());
        u0->write(idx, make_float2(0.f, 0.f));
        u1->write(idx, make_float2(0.f, 0.f));

        rho0->write(idx, 0.f);
        rho1->write(idx, 0.f);

        p0->write(idx, 0.f);
        p1->write(idx, 0.f);
        div->write(idx, 0.f);
    });

    Shader2D<> clear_pressure = device.compile<2>([&] {
        UInt idx = index(dispatch_id().xy());
        p0->write(idx, 0.f);
        p1->write(idx, 0.f);
    });

    auto substep = [&](CommandList &cmd_list) noexcept {
        auto angle = std::sin(sim_time * 4.0f) * 1.5f;
        float2 vel = float2(std::cos(angle) * speed, std::sin(angle) * speed);

        cmd_list << init(rho0, u0, 5, vel).dispatch(n_grid, n_grid) // update emitters
                 << integrate(u0, rho0, dt).dispatch(n_grid, n_grid)// force integrate
                 << divergence(u0, div).dispatch(n_grid, n_grid);

        // pressure solve
        cmd_list << clear_pressure().dispatch(n_grid, n_grid);
        for (int j = 0; j < iterations; ++j) {
            cmd_list << pressure_solve(p0, p1, div).dispatch(n_grid, n_grid);
            std::swap(p0, p1);
        }

        cmd_list << pressure_apply(p0, u0).dispatch(n_grid, n_grid)         // velocity update
                 << advect(u0, u1, rho0, rho1, dt).dispatch(n_grid, n_grid);// semi-Lagrangian advection

        // swap buffers
        std::swap(u0, u1);
        std::swap(rho0, rho1);

        sim_time += dt;
    };

    Shader2D<> draw_rho = device.compile<2>([&] {
        UInt2 coord = dispatch_id().xy();
        auto ij = index(coord);
        auto value = rho0->read(ij);
        display->write(make_uint2(coord.x, n_grid - 1 - coord.y), make_float4(value, 0.f, 0.f, 1.f));
    });

    stream << init_grid().dispatch(n_grid, n_grid)
           << hash_grid.reserve(100)
           << synchronize();
    while (!window.should_close()) {
        CommandList cmd_list;
        for (uint i = 0u; i < sim_substeps; i++) { substep(cmd_list); }
        cmd_list << draw_rho().dispatch(n_grid, n_grid);
        stream << cmd_list.commit() << swap_chain.present(display);
        window.poll_events();
    }
    stream << synchronize();
}
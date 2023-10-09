//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "utils/status_util.h"
#include "runtime/buffer.h"
#include "runtime/stream.h"
#include "runtime/ext/metal/metal_command.h"
#include <spdlog/fmt/fmt.h>

#include "benchmark_api.h"

struct ShaderCode {
    const char *name;// Test case name
    metal::MetalCommand::ArithmeticMode op;
};

static ShaderCode kShaderCodeCases[] = {
//    {"add/loop", metal::MetalCommand::ArithmeticMode::Add},
    {"add/intrinsic", metal::MetalCommand::ArithmeticMode::Add},
//    {"mul/loop", metal::MetalCommand::ArithmeticMode::Mul},
    {"mul/intrinsic", metal::MetalCommand::ArithmeticMode::Mul},
};

namespace luisa {
static void calculate_simd_group_arithmetic(::benchmark::State &state,
                                            LatencyMeasureMode mode,
                                            Device *device,
                                            int num_elements, uint32_t subgroup_size,
                                            metal::MetalCommand::ArithmeticMode arith_op) {
    auto stream = device->create_stream();
    //===-------------------------------------------------------------------===/
    // Create buffers
    //===-------------------------------------------------------------------===/
    size_t buffer_num_bytes = num_elements * sizeof(float);

    auto src_buffer = device->create_buffer<float>(num_elements);
    auto dst_buffer = device->create_buffer<float>(num_elements);
    auto command = metal::MetalCommand::simd_group_arithmetic(src_buffer.view(), dst_buffer.view(), num_elements, arith_op);
    command->alloc_pso(device);

    //===-------------------------------------------------------------------===/
    // Set source buffer data
    //===-------------------------------------------------------------------===/
    // +: fill the whole buffer as 1.0f.
    // *: fill with alternating subgroup_size and (1 / subgroup_size).
    auto ptr = malloc(buffer_num_bytes);
    auto *src_float_buffer = reinterpret_cast<float *>(ptr);
    switch (arith_op) {
        case metal::MetalCommand::ArithmeticMode::Add: {
            for (int i = 0; i < buffer_num_bytes / sizeof(float); ++i) {
                src_float_buffer[i] = 1.0f;
            }
        } break;
        case metal::MetalCommand::ArithmeticMode::Mul: {
            for (int i = 0; i < buffer_num_bytes / sizeof(float); i += 2) {
                src_float_buffer[i] = subgroup_size;
                src_float_buffer[i + 1] = 1.0f / subgroup_size;
            }
        } break;
    }
    stream << src_buffer.copy_from(ptr) << synchronize();
    free(ptr);

    //===-------------------------------------------------------------------===/
    // Dispatch
    //===-------------------------------------------------------------------===/
    {
        stream << command->clone()
               << synchronize();
    }

    //===-------------------------------------------------------------------===/
    // Verify destination buffer data
    //===-------------------------------------------------------------------===/
    ptr = malloc(buffer_num_bytes);
    stream << dst_buffer.copy_to(ptr) << synchronize();

    auto *dst_float_buffer = reinterpret_cast<float *>(ptr);
    switch (arith_op) {
        case metal::MetalCommand::ArithmeticMode::Add: {
            for (int i = 0; i < buffer_num_bytes / sizeof(float); ++i) {
                float expected_value = 1.0f;
                if (i % subgroup_size == 0) {
                    expected_value = subgroup_size;
                }

                BM_CHECK_EQ(dst_float_buffer[i], expected_value)
                    << fmt::format("destination buffer element #{} has incorrect value: expected to be {} but found {}",
                                   i, expected_value, dst_float_buffer[i]);
            }
        } break;
        case metal::MetalCommand::ArithmeticMode::Mul: {
            for (int i = 0; i < buffer_num_bytes / sizeof(float); ++i) {
                float expected_value = 0.0f;
                if (i % subgroup_size == 0) {
                    expected_value = 1.0f;
                } else if (i % 2 == 0) {
                    expected_value = subgroup_size;
                } else {
                    expected_value = 1.0f / subgroup_size;
                }

                BM_CHECK_EQ(dst_float_buffer[i], expected_value)
                    << fmt::format("destination buffer element #{} has incorrect value: expected to be {} but found {}",
                                   i, expected_value, dst_float_buffer[i]);
            }
        } break;
    }
    free(ptr);

    //===-------------------------------------------------------------------===/
    // Benchmarking
    //===-------------------------------------------------------------------===/
    for ([[maybe_unused]] auto _ : state) {
        auto start_time = std::chrono::high_resolution_clock::now();
        stream << command->clone()
               << synchronize();
        auto end_time = std::chrono::high_resolution_clock::now();

        auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
        switch (mode) {
            case LatencyMeasureMode::kSystemSubmit:
                state.SetIterationTime(elapsed_seconds.count());
                break;
        }
    }

    state.counters["FLOps"] =
        ::benchmark::Counter(num_elements,
                             ::benchmark::Counter::kIsIterationInvariant |
                                 ::benchmark::Counter::kIsRate,
                             ::benchmark::Counter::kIs1000);
}

void SimdGroupArithmetic::register_benchmarks(Device &device, LatencyMeasureMode mode) {
    const auto gpu_name = device.backend_name();

    static int kBufferNumElements = 1 << 20;// 1M

    for (const auto &shader : kShaderCodeCases) {// Loop/intrinsic shader
        std::string test_name = fmt::format("{}/{}/{}", gpu_name, shader.name, kBufferNumElements);
        ::benchmark::RegisterBenchmark(test_name, calculate_simd_group_arithmetic, mode, &device,
                                       kBufferNumElements, device.compute_warp_size(), shader.op)
            ->UseManualTime()
            ->Unit(::benchmark::kMicrosecond);
    }
}
}// namespace luisa
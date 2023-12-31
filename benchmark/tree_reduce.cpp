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
    const char *name;     // Test case name
    size_t batch_elements;// Number of elements in each batch
    metal::MetalCommand::ReduceMode mode;
    bool is_integer;// Whether the elements should be integers
};

#define FLOAT_SHADER_CASE(kind, size, mode)      \
    {                                            \
        #kind "/batch=" #size, size, mode, false \
    }

#define INT_SHADER_CASE(kind, size, mode)       \
    {                                           \
        #kind "/batch=" #size, size, mode, true \
    }

static ShaderCode kShaders[] = {
    FLOAT_SHADER_CASE(loop, 16, metal::MetalCommand::ReduceMode::Loop),
    FLOAT_SHADER_CASE(loop, 32, metal::MetalCommand::ReduceMode::Loop),
    FLOAT_SHADER_CASE(loop, 64, metal::MetalCommand::ReduceMode::Loop),
    FLOAT_SHADER_CASE(loop, 128, metal::MetalCommand::ReduceMode::Loop),
    FLOAT_SHADER_CASE(subgroup, 16, metal::MetalCommand::ReduceMode::SimdGroup),
    FLOAT_SHADER_CASE(subgroup, 32, metal::MetalCommand::ReduceMode::SimdGroup),
    FLOAT_SHADER_CASE(subgroup, 64, metal::MetalCommand::ReduceMode::SimdGroup),
    FLOAT_SHADER_CASE(subgroup, 128, metal::MetalCommand::ReduceMode::SimdGroup),

    INT_SHADER_CASE(loop, 16, metal::MetalCommand::ReduceMode::Loop),
    INT_SHADER_CASE(loop, 32, metal::MetalCommand::ReduceMode::Loop),
    INT_SHADER_CASE(loop, 64, metal::MetalCommand::ReduceMode::Loop),
    INT_SHADER_CASE(loop, 128, metal::MetalCommand::ReduceMode::Loop),
    INT_SHADER_CASE(subgroup, 16, metal::MetalCommand::ReduceMode::SimdGroup),
    INT_SHADER_CASE(subgroup, 32, metal::MetalCommand::ReduceMode::SimdGroup),
    INT_SHADER_CASE(subgroup, 64, metal::MetalCommand::ReduceMode::SimdGroup),
    INT_SHADER_CASE(subgroup, 128, metal::MetalCommand::ReduceMode::SimdGroup),
};

namespace luisa {
static void reduce(::benchmark::State &state,
                   LatencyMeasureMode mode,
                   Device *device,
                   size_t total_elements,
                   const ShaderCode &shader) {
    auto stream = device->create_stream();
    //===-------------------------------------------------------------------===/
    // Create buffers
    //===-------------------------------------------------------------------===/
    const size_t buffer_size = total_elements * sizeof(float);

    auto reduce_buffer = device->create_buffer<float>(total_elements);
    auto data_buffer = device->create_buffer<float>(total_elements);
    auto command = metal::MetalCommand::tree_reduce(reduce_buffer.view(), shader.batch_elements, shader.mode, shader.is_integer);
    command->alloc_pso(device);

    //===-------------------------------------------------------------------===/
    // Set source buffer data
    //===-------------------------------------------------------------------===/
    auto generate_float_data = [](size_t i) -> float { return float(i % 9 - 4) * 0.5f; };
    auto generate_int_data = [](size_t i) -> int { return int(i % 13 - 7); };

    auto ptr = malloc(buffer_size);
    if (shader.is_integer) {
        auto *src_int_buffer = reinterpret_cast<int *>(ptr);
        for (size_t i = 0; i < buffer_size / sizeof(int); i++) {
            src_int_buffer[i] = generate_int_data(i);
        }
    } else {
        auto *src_float_buffer = reinterpret_cast<float *>(ptr);
        for (size_t i = 0; i < buffer_size / sizeof(float); i++) {
            src_float_buffer[i] = generate_float_data(i);
        }
    }
    stream << data_buffer.copy_from(ptr) << synchronize();
    free(ptr);

    //===-------------------------------------------------------------------===/
    // Dispatch
    //===-------------------------------------------------------------------===/
    {
        stream << reduce_buffer.copy_from(data_buffer)
               << command->clone()
               << synchronize();
    }

    //===-------------------------------------------------------------------===/
    // Verify destination buffer data
    //===-------------------------------------------------------------------===/
    ptr = malloc(buffer_size);
    stream << reduce_buffer.copy_to(ptr) << synchronize();
    if (shader.is_integer) {
        auto *dst_int_buffer = reinterpret_cast<int *>(ptr);
        int total = 0;
        for (size_t i = 0; i < total_elements; i++) {
            total += generate_int_data(i);
        };
        BM_CHECK_EQ(dst_int_buffer[0], total)
            << fmt::format("destination buffer element #0 has incorrect value: expected to be {} but found {}",
                           total, dst_int_buffer[0]);
    } else {
        auto *dst_float_buffer = reinterpret_cast<float *>(ptr);
        float total = 0.f;
        for (size_t i = 0; i < total_elements; i++) {
            total += generate_float_data(i);
        };
        BM_CHECK_FLOAT_EQ(dst_float_buffer[0], total, 0.01f)
            << fmt::format("destination buffer element #0 has incorrect value: expected to be {} but found {}",
                           total, dst_float_buffer[0]);
    }
    free(ptr);

    //===-------------------------------------------------------------------===/
    // Benchmarking
    //===-------------------------------------------------------------------===/
    for ([[maybe_unused]] auto _ : state) {
        stream << reduce_buffer.copy_from(data_buffer) << synchronize();

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

    state.SetBytesProcessed(state.iterations() * buffer_size);
    state.counters["FLOps"] =
        ::benchmark::Counter(total_elements,
                             ::benchmark::Counter::kIsIterationInvariant |
                                 ::benchmark::Counter::kIsRate,
                             ::benchmark::Counter::kIs1000);
}

void TreeReduce::register_benchmarks(Device &device, LatencyMeasureMode mode) {
    const auto gpu_name = device.backend_name();

    for (const auto &shader : kShaders) {
        // Find the power of batch_elements that are larger than 1M.
        size_t total_elements = shader.batch_elements;
        while (total_elements < (1 << 20)) total_elements *= shader.batch_elements;

        std::string test_name = fmt::format("{}/{}{}{}", gpu_name, total_elements, (shader.is_integer ? "xi32/" : "xf32/"), shader.name);

        ::benchmark::RegisterBenchmark(test_name, reduce, mode, &device,
                                       total_elements, shader)
            ->UseManualTime()
            ->Unit(::benchmark::kMicrosecond);
    }
}
}// namespace luisa
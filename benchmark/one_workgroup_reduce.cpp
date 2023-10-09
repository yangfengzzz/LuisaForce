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
    const char *name;  // Test case name
    int workgroup_size;// Number of invocations per workgroup
    metal::MetalCommand::ReduceMode mode;
};

#define ATOMIC_CASE(size) \
    { "atomic", size, metal::MetalCommand::ReduceMode::Atomic }

static ShaderCode kShaders[] = {
    {"loop", 16, metal::MetalCommand::ReduceMode::Loop},
    {"subgroup", 16, metal::MetalCommand::ReduceMode::SimdGroup},
    ATOMIC_CASE(16),
    ATOMIC_CASE(32),
    ATOMIC_CASE(64),
    ATOMIC_CASE(128),
    ATOMIC_CASE(256),
};

namespace luisa {
static void reduce(::benchmark::State &state,
                   LatencyMeasureMode mode,
                   Device *device,
                   size_t total_elements,
                   const ShaderCode& shader) {
    auto stream = device->create_stream();
    //===-------------------------------------------------------------------===/
    // Create buffers
    //===-------------------------------------------------------------------===/
    const size_t src_buffer_size = total_elements * sizeof(float);
    const size_t dst_buffer_size = sizeof(float);

    auto src_buffer = device->create_buffer<float>(total_elements);
    auto dst_buffer = device->create_buffer<float>(1);
    auto command = metal::MetalCommand::one_workgroup_reduce(src_buffer.view(), dst_buffer.view(), total_elements, shader.mode);
    command->alloc_pso(device);

    //===-------------------------------------------------------------------===/
    // Set source buffer data
    //===-------------------------------------------------------------------===/
    auto generate_float_data = [](size_t i) { return float(i % 9 - 4) * 0.5f; };

    auto ptr = malloc(src_buffer_size);
    auto *src_float_buffer = reinterpret_cast<float *>(ptr);
    for (size_t i = 0; i < src_buffer_size / sizeof(float); i++) {
        src_float_buffer[i] = generate_float_data(i);
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
    ptr = malloc(dst_buffer_size);
    stream << dst_buffer.copy_to(ptr) << synchronize();

    auto *dst_float_buffer = reinterpret_cast<float *>(ptr);
    float total = 0.f;
    for (size_t i = 0; i < total_elements; i++) {
        total += generate_float_data(i);
    };
    BM_CHECK_FLOAT_EQ(dst_float_buffer[0], total, 0.01f)
        << fmt::format("destination buffer element #0 has incorrect value: expected to be {} but found {}",
                       total, dst_float_buffer[0]);
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

    state.SetBytesProcessed(state.iterations() * src_buffer_size);
    state.counters["FLOps"] =
        ::benchmark::Counter(total_elements,
                             ::benchmark::Counter::kIsIterationInvariant |
                                 ::benchmark::Counter::kIsRate,
                             ::benchmark::Counter::kIs1000);
}

void OneWorkgroupReduce::register_benchmarks(Device &device, LatencyMeasureMode mode) {
    const auto gpu_name = device.backend_name();

    for (const auto &shader : kShaders) {
        for (size_t total_elements : {1 << 10, 1 << 12, 1 << 14, 1 << 16}) {
            std::string test_name = fmt::format("{}/#elements={}/workgroup_size={}/{}",
                                                gpu_name, total_elements,
                                                shader.workgroup_size, shader.name);

            ::benchmark::RegisterBenchmark(test_name, reduce, mode, &device,
                                           total_elements, shader)
                ->UseManualTime()
                ->Unit(::benchmark::kMicrosecond);
        }
    }
}
}// namespace luisa
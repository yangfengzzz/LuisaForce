//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "benchmark_api.h"
#include "utils/data_type_util.h"
#include "utils/status_util.h"
#include "runtime/buffer.h"
#include "runtime/stream.h"
#ifdef LUISA_PLATFORM_APPLE
#include "runtime/ext/metal/metal_command.h"
#include "runtime/ext/debug_capture_ext.h"
#endif

#ifdef LUISA_PLATFORM_CUDA
#include "runtime/ext/cuda/cuda_commands.h"
#endif

#include <spdlog/fmt/fmt.h>

namespace luisa {
static void throughput(::benchmark::State &state,
                       LatencyMeasureMode mode,
                       Device *device,
                       size_t num_element, int loop_count, DataType data_type) {
    auto stream = device->create_stream();
    //===-------------------------------------------------------------------===/
    // Create buffers
    //===-------------------------------------------------------------------===/
    auto src0_buffer = device->create_buffer<float>(num_element);
    auto src1_buffer = device->create_buffer<float>(num_element);
    auto dst_buffer = device->create_buffer<float>(num_element);
#ifdef LUISA_PLATFORM_APPLE
    auto command = metal::MetalCommand::mad_throughput(src0_buffer.view(), src1_buffer.view(), dst_buffer.view());
    command->alloc_pso(device);
#endif

#ifdef LUISA_PLATFORM_CUDA
    auto command = cuda::CudaCommand::mad_throughput(src0_buffer.view(), src1_buffer.view(), dst_buffer.view());
#endif

    //===-------------------------------------------------------------------===/
    // Set source buffer data
    //===-------------------------------------------------------------------===/
    auto getSrc0 = [](size_t i) {
        float v = float((i % 9) + 1) * 0.1f;
        return v;
    };
    auto getSrc1 = [](size_t i) {
        float v = float((i % 5) + 1) * 1.f;
        return v;
    };

    if (data_type == DataType::fp16) {
        std::vector<uint16_t> src_float_buffer(num_element);
        for (size_t i = 0; i < num_element; i++) {
            src_float_buffer[i] = fp16(getSrc0(i)).get_value();
        }
        stream << src0_buffer.copy_from(src_float_buffer.data()) << synchronize();

        for (size_t i = 0; i < num_element; i++) {
            src_float_buffer[i] = fp16(getSrc1(i)).get_value();
        }
        stream << src1_buffer.copy_from(src_float_buffer.data()) << synchronize();
    } else if (data_type == DataType::fp32) {
        std::vector<float> src_float_buffer(num_element);
        for (size_t i = 0; i < num_element; i++) {
            src_float_buffer[i] = getSrc0(i);
        }
        stream << src0_buffer.copy_from(src_float_buffer.data()) << synchronize();

        for (size_t i = 0; i < num_element; i++) {
            src_float_buffer[i] = getSrc1(i);
        }
        stream << src1_buffer.copy_from(src_float_buffer.data()) << synchronize();
    }

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

    if (data_type == DataType::fp16) {
        std::vector<uint16_t> dst_float_buffer(num_element);
        stream << dst_buffer.copy_to(dst_float_buffer.data()) << synchronize();
        for (size_t i = 0; i < num_element; i++) {
            float limit = getSrc1(i) * (1.f / (1.f - getSrc0(i)));
            BM_CHECK_FLOAT_EQ(fp16(dst_float_buffer[i]).to_float(), limit, 0.5f)
                << "destination buffer element #" << i
                << " has incorrect value: expected to be " << limit
                << " but found " << fp16(dst_float_buffer[i]).to_float();
        }
    } else if (data_type == DataType::fp32) {
        std::vector<float> dst_float_buffer(num_element);
        stream << dst_buffer.copy_to(dst_float_buffer.data()) << synchronize();
        for (size_t i = 0; i < num_element; i++) {
            float limit = getSrc1(i) * (1.f / (1.f - getSrc0(i)));
            BM_CHECK_FLOAT_EQ(dst_float_buffer[i], limit, 0.01f)
                << fmt::format("destination buffer element #{}, has incorrect value: expected to be {} but found {}",
                               i, limit, dst_float_buffer[i]);
        }
    }

    //===-------------------------------------------------------------------===/
    // Benchmarking
    //===-------------------------------------------------------------------===/
    {
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
        double numOperation = double(num_element) * 2. /*fma*/ *
                              10. /*10 elements per loop iteration*/ *
                              double(loop_count);
        state.counters["FLOps"] =
            ::benchmark::Counter(numOperation,
                                 ::benchmark::Counter::kIsIterationInvariant |
                                     ::benchmark::Counter::kIsRate,
                                 ::benchmark::Counter::kIs1000);
    }
}

void MADThroughPut::register_benchmarks(Device &device, LatencyMeasureMode mode) {
    const auto gpu_name = device.backend_name();

    const size_t num_element = 1024 * 1024;
    const int min_loop_count = 100000;
    const int max_loop_count = min_loop_count * 2;

    for (int loop_count = min_loop_count; loop_count <= max_loop_count; loop_count += min_loop_count) {
        std::string test_name = fmt::format("{}/{}/{}/{}", gpu_name, "mad_throughput", num_element, loop_count);

        ::benchmark::RegisterBenchmark(test_name, throughput, mode, &device,
                                       num_element, loop_count, DataType::fp32)
            ->UseManualTime()
            ->Unit(::benchmark::kMicrosecond)
            ->MinTime(std::numeric_limits<float>::epsilon());// use cache make calculation fast after warmup
    }
}

}// namespace luisa

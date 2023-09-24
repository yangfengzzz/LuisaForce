//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "mad_throughput.h"
#include "utils/data_type_util.h"
#include "utils/status_util.h"
#include "runtime/buffer.h"
#include "runtime/stream.h"
#include "runtime/ext/metal/metal_command.h"
#include <spdlog/fmt/fmt.h>

namespace vox::benchmark {
static void throughput(::benchmark::State &state,
                       LatencyMeasureMode mode,
                       Device *device,
                       size_t num_element, int loop_count, compute::DataType data_type) {
    auto stream = device->create_stream();
    //===-------------------------------------------------------------------===/
    // Create buffers
    //===-------------------------------------------------------------------===/
    auto src0_buffer = device->create_buffer<float>(num_element);
    auto src1_buffer = device->create_buffer<float>(num_element);
    auto dst_buffer = device->create_buffer<float>(num_element);

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

    if (data_type == compute::DataType::fp16) {
        std::vector<uint16_t> src_float_buffer(num_element);
        for (size_t i = 0; i < num_element; i++) {
            src_float_buffer[i] = compute::fp16(getSrc0(i)).get_value();
        }
        stream << src0_buffer.copy_from(src_float_buffer.data()) << synchronize();

        for (size_t i = 0; i < num_element; i++) {
            src_float_buffer[i] = compute::fp16(getSrc1(i)).get_value();
        }
        stream << src0_buffer.copy_from(src_float_buffer.data()) << synchronize();
    } else if (data_type == compute::DataType::fp32) {
        std::vector<float> src_float_buffer(num_element);
        for (size_t i = 0; i < num_element; i++) {
            src_float_buffer[i] = getSrc0(i);
        }
        stream << src0_buffer.copy_from(src_float_buffer.data()) << synchronize();

        for (size_t i = 0; i < num_element; i++) {
            src_float_buffer[i] = getSrc1(i);
        }
        stream << src0_buffer.copy_from(src_float_buffer.data()) << synchronize();
    }

    //===-------------------------------------------------------------------===/
    // Dispatch
    //===-------------------------------------------------------------------===/
    {
        stream << metal::MetalCommand::mad_throughput(src0_buffer.view(), src1_buffer.view(), dst_buffer.view())
               << synchronize();
    }

    //===-------------------------------------------------------------------===/
    // Verify destination buffer data
    //===-------------------------------------------------------------------===/

    if (data_type == compute::DataType::fp16) {
        std::vector<uint16_t> dst_float_buffer(num_element);
        stream << dst_buffer.copy_to(dst_float_buffer.data()) << synchronize();
        for (size_t i = 0; i < num_element; i++) {
            float limit = getSrc1(i) * (1.f / (1.f - getSrc0(i)));
            BM_CHECK_FLOAT_EQ(compute::fp16(dst_float_buffer[i]).to_float(), limit, 0.5f)
                << "destination buffer element #" << i
                << " has incorrect value: expected to be " << limit
                << " but found " << compute::fp16(dst_float_buffer[i]).to_float();
        }
    } else if (data_type == compute::DataType::fp32) {
        std::vector<float> dst_float_buffer(num_element);
        stream << dst_buffer.copy_to(dst_float_buffer.data()) << synchronize();
        for (size_t i = 0; i < num_element; i++) {
            float limit = getSrc1(i) * (1.f / (1.f - getSrc0(i)));
            BM_CHECK_FLOAT_EQ(dst_float_buffer[i], limit, 0.01f)
                << "destination buffer element #" << i
                << " has incorrect value: expected to be " << limit
                << " but found " << dst_float_buffer[i];
        }
    }

    //===-------------------------------------------------------------------===/
    // Benchmarking
    //===-------------------------------------------------------------------===/
    {
        for ([[maybe_unused]] auto _ : state) {
            // auto scope = compute::create_capture_scope("test", *device);
            // scope->beginScope();

            auto start_time = std::chrono::high_resolution_clock::now();
            stream << metal::MetalCommand::mad_throughput(src0_buffer.view(), src1_buffer.view(), dst_buffer.view())
                   << synchronize();
            auto end_time = std::chrono::high_resolution_clock::now();

            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time);
            switch (mode) {
                case LatencyMeasureMode::kSystemSubmit:
                    state.SetIterationTime(elapsed_seconds.count());
                    break;
            }

            // scope->endScope();
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
                                       num_element, loop_count, compute::DataType::fp32)
            ->UseManualTime()
            ->Unit(::benchmark::kMicrosecond)
            ->MinTime(std::numeric_limits<float>::epsilon());// use cache make calculation fast after warmup
    }
}

}// namespace vox::benchmark
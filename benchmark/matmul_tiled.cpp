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

#include "matmul_tiled.h"

namespace vox::benchmark {
static void matmul(::benchmark::State &state,
                       LatencyMeasureMode mode,
                       Device *device,
                       size_t num_element, int loop_count, compute::DataType data_type) {

}

void MatMul::register_benchmarks(Device &device, LatencyMeasureMode mode) {
    const auto gpu_name = device.backend_name();

    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    const size_t num_element = 1024 * 1024;
    const int min_loop_count = 100000;
    const int max_loop_count = min_loop_count * 2;

    for (int loop_count = min_loop_count; loop_count <= max_loop_count; loop_count += min_loop_count) {
        std::string test_name = fmt::format("{}/{}/{}/{}", gpu_name, "matmul", num_element, loop_count);

        ::benchmark::RegisterBenchmark(test_name, matmul, mode, &device,
                                       num_element, loop_count, compute::DataType::fp32)
            ->UseManualTime()
            ->Unit(::benchmark::kMicrosecond)
            ->MinTime(std::numeric_limits<float>::epsilon());// use cache make calculation fast after warmup
    }
}

}// namespace vox::benchmark

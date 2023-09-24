//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <benchmark/benchmark.h>
#include "runtime/device.h"

using namespace luisa;
using namespace luisa::compute;

namespace vox {
enum class LatencyMeasureMode {
    // time spent from queue submit to returning from queue wait
    kSystemSubmit,
};

class BenchmarkAPI {
public:
    // Registers all Vulkan benchmarks for the current benchmark binary.
    //
    // The |overhead_seconds| field in |latency_measure| should subtracted from the
    // latency measured by the registered benchmarks for
    // LatencyMeasureMode::kSystemDispatch.
    virtual void register_benchmarks(Device &queue, LatencyMeasureMode mode) = 0;
};
}// namespace vox
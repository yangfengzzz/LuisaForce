//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "utils/data_type_util.h"
#include "utils/status_util.h"
#include "runtime/buffer.h"
#include "runtime/stream.h"
#include "runtime/ext/metal/metal_command.h"
#include <spdlog/fmt/fmt.h>

#include "benchmark_api.h"

namespace luisa {
void OneWorkgroupReduce::register_benchmarks(Device &device, LatencyMeasureMode mode) {
    const auto gpu_name = device.backend_name();
}
}// namespace luisa
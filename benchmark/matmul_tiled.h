//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "benchmark_api.h"

namespace vox::benchmark {
class MatMul : public BenchmarkAPI {
public:
    void register_benchmarks(Device &device, LatencyMeasureMode mode) override;
};

}// namespace vox::benchmark
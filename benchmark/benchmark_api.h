//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <benchmark/benchmark.h>
#include "runtime/device.h"

using namespace luisa::compute;

namespace luisa {
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

class MADThroughPut : public BenchmarkAPI {
public:
    void register_benchmarks(Device &device, LatencyMeasureMode mode) override;
};

class MatMul : public BenchmarkAPI {
public:
    void register_benchmarks(Device &device, LatencyMeasureMode mode) override;
};

// matrix-matrix transposed multiplication of two 2D inputs.
class MMT : public BenchmarkAPI {
public:
    void register_benchmarks(Device &device, LatencyMeasureMode mode) override;
};

class AtomicReduce : public BenchmarkAPI {
public:
    void register_benchmarks(Device &device, LatencyMeasureMode mode) override;
};

class OneWorkgroupReduce : public BenchmarkAPI {
public:
    void register_benchmarks(Device &device, LatencyMeasureMode mode) override;
};

class TreeReduce : public BenchmarkAPI {
public:
    void register_benchmarks(Device &device, LatencyMeasureMode mode) override;
};

class SimdGroupArithmetic : public BenchmarkAPI {
public:
    void register_benchmarks(Device &device, LatencyMeasureMode mode) override;
};

}// namespace luisa
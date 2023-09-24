//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <benchmark/benchmark.h>
#include "runtime/context.h"
#include "runtime/device.h"

#include "mad_throughput.h"

using namespace luisa;
using namespace luisa::compute;

int main(int argc, char **argv) {
    ::benchmark::Initialize(&argc, argv);
    Context context{argv[0]};
    Device device = context.create_device();

    vox::LatencyMeasureMode mode = vox::LatencyMeasureMode::kSystemSubmit;
    auto app = std::make_unique<vox::benchmark::MADThroughPut>();
    app->register_benchmarks(device, mode);

    ::benchmark::RunSpecifiedBenchmarks();
}
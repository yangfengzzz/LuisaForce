//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <iostream>
#include <random>
#include <numeric>
#include "core/logging.h"
#include "runtime/device.h"
#include "runtime/context.h"
#include "runtime/image.h"
#include "runtime/stream.h"
#include "runtime/buffer.h"
#include "dsl/syntax.h"
#include "runtime/ext/cuda/lcub/device_scan.h"

using namespace luisa;
using namespace luisa::compute;
using namespace luisa::compute::cuda::lcub;

void device_scan_test(Device &device, std::vector<int> &gt, std::vector<int> &out) {
    // host
    constexpr auto num_item = 100;
    std::vector<int> input(num_item);
    std::iota(input.begin(), input.end(), 0);

    gt.resize(num_item);
    out.resize(num_item);

    std::exclusive_scan(input.begin(), input.end(), gt.begin(), 0);

    // device
    auto stream = device.create_stream();
    auto d_in = device.create_buffer<int>(num_item);
    stream << d_in.copy_from(input.data());
    auto d_out = device.create_buffer<int>(num_item);

    Buffer<int> temp_storage;
    size_t temp_storage_size = -1;
    DeviceScan::ExclusiveSum(temp_storage_size, d_in, d_out, num_item);
    temp_storage = device.create_buffer<int>(temp_storage_size);
    stream << DeviceScan::ExclusiveSum(temp_storage, d_in, d_out, num_item);
    stream << d_out.copy_to(out.data()) << synchronize();

	// check
    for (int i = 0; i < num_item; i++) {
        if (gt[i] != out[i]) {
            std::cout << "Error: " << i << " " << gt[i] << " " << out[i] << std::endl;
            return;
        }
    }
    std::cout << "===Success===" << std::endl;
}

int main(int argc, char *argv[]) {
	// cuda only
    Context context{argv[0]};
    Device device = context.create_device();
    std::vector<int> gt, out;
    device_scan_test(device, gt, out);
}


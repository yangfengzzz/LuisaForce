//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "cuda_util.h"
#include "cuda_context.h"
#include "scan.h"

#include <cub/device/device_scan.cuh>

namespace luisa::compute::cuda {
template<typename T>
void scan_device(const T *values_in, T *values_out, int n, bool inclusive, CUstream stream) {
    ContextGuard guard(cuda_context_get_current());

    // compute temporary memory required
    size_t scan_temp_size;
    if (inclusive) {
        check_cuda(cub::DeviceScan::InclusiveSum(nullptr, scan_temp_size, values_in, values_out, n));
    } else {
        check_cuda(cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_size, values_in, values_out, n));
    }

    void *temp_buffer = alloc_temp_device(WP_CURRENT_CONTEXT, scan_temp_size, stream);

    // scan
    if (inclusive) {
        check_cuda(cub::DeviceScan::InclusiveSum(temp_buffer, scan_temp_size, values_in, values_out, n, stream));
    } else {
        check_cuda(cub::DeviceScan::ExclusiveSum(temp_buffer, scan_temp_size, values_in, values_out, n, stream));
    }

    free_temp_device(WP_CURRENT_CONTEXT, temp_buffer, stream);
}

template void scan_device(const int *, int *, int, bool, CUstream stream);
template void scan_device(const float *, float *, int, bool, CUstream stream);

void array_scan_int_device(uint64_t in, uint64_t out, int len, bool inclusive, CUstream stream) {
    scan_device((const int *)in, (int *)out, len, inclusive, stream);
}

void array_scan_float_device(uint64_t in, uint64_t out, int len, bool inclusive, CUstream stream) {
    scan_device((const float *)in, (float *)out, len, inclusive, stream);
}

}// namespace luisa::compute::cuda
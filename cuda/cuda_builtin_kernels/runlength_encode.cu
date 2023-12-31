//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "cuda_context.h"
#include "cuda_util.h"

#include <cub/device/device_run_length_encode.cuh>

namespace luisa::compute::cuda {
template<typename T>
void runlength_encode_device(int n,
                             const T *values,
                             T *run_values,
                             int *run_lengths,
                             int *run_count, CUstream stream) {
    ContextGuard guard(cuda_context_get_current());

    size_t buff_size = 0;
    check_cuda(cub::DeviceRunLengthEncode::Encode(
        nullptr, buff_size, values, run_values, run_lengths, run_count,
        n, stream));

    void *temp_buffer = alloc_temp_device(WP_CURRENT_CONTEXT, buff_size, stream);

    check_cuda(cub::DeviceRunLengthEncode::Encode(
        temp_buffer, buff_size, values, run_values, run_lengths, run_count,
        n, stream));

    free_temp_device(WP_CURRENT_CONTEXT, temp_buffer, stream);
}

void runlength_encode_int_device(uint64_t values,
                                 uint64_t run_values,
                                 uint64_t run_lengths,
                                 uint64_t run_count,
                                 int n, CUstream stream) {
    return runlength_encode_device<int>(
        n,
        reinterpret_cast<const int *>(values),
        reinterpret_cast<int *>(run_values),
        reinterpret_cast<int *>(run_lengths),
        reinterpret_cast<int *>(run_count), stream);
}

}// namespace luisa::compute::cuda
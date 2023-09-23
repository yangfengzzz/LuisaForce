//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "device_partition.h"
#include "dcub_utils.cuh"
#include <cub/device/device_partition.cuh>

namespace luisa::compute::cuda::dcub {
// DOC:  https://nvlabs.github.io/cub/structcub_1_1_device_partition.html
cudaError_t DevicePartition::Flagged(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_in, const int32_t *d_flags, int32_t *d_out, int32_t *d_num_selected_out, int num_items, cudaStream_t stream) {
    return ::cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items, stream);
}

cudaError_t DevicePartition::Flagged(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_in, const int32_t *d_flags, uint32_t *d_out, int32_t *d_num_selected_out, int num_items, cudaStream_t stream) {
    return ::cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items, stream);
}

cudaError_t DevicePartition::Flagged(void *d_temp_storage, size_t &temp_storage_bytes, const int64_t *d_in, const int32_t *d_flags, int64_t *d_out, int32_t *d_num_selected_out, int num_items, cudaStream_t stream) {
    return ::cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items, stream);
}

cudaError_t DevicePartition::Flagged(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_in, const int32_t *d_flags, uint64_t *d_out, int32_t *d_num_selected_out, int num_items, cudaStream_t stream) {
    return ::cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items, stream);
}

cudaError_t DevicePartition::Flagged(void *d_temp_storage, size_t &temp_storage_bytes, const float *d_in, const int32_t *d_flags, float *d_out, int32_t *d_num_selected_out, int num_items, cudaStream_t stream) {
    return ::cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items, stream);
}

cudaError_t DevicePartition::Flagged(void *d_temp_storage, size_t &temp_storage_bytes, const double *d_in, const int32_t *d_flags, double *d_out, int32_t *d_num_selected_out, int num_items, cudaStream_t stream) {
    return ::cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items, stream);
}
}// namespace luisa::compute::cuda::dcub

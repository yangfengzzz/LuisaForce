//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "runtime/ext/cuda/lcub/dcub_common.h"

namespace luisa::compute::cuda::dcub {

class DCUB_API DeviceSelect {
    // DOC:  https://nvlabs.github.io/cub/structcub_1_1_device_select.html
public:
    static cudaError_t Flagged(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_in, const int32_t *d_flags, int32_t *d_out, int32_t *d_num_selected_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Flagged(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_in, const int32_t *d_flags, uint32_t *d_out, int32_t *d_num_selected_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Flagged(void *d_temp_storage, size_t &temp_storage_bytes, const int64_t *d_in, const int32_t *d_flags, int64_t *d_out, int32_t *d_num_selected_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Flagged(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_in, const int32_t *d_flags, uint64_t *d_out, int32_t *d_num_selected_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Flagged(void *d_temp_storage, size_t &temp_storage_bytes, const float *d_in, const int32_t *d_flags, float *d_out, int32_t *d_num_selected_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Flagged(void *d_temp_storage, size_t &temp_storage_bytes, const double *d_in, const int32_t *d_flags, double *d_out, int32_t *d_num_selected_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Unique(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_in, int32_t *d_out, int32_t *d_num_selected_out, int num_items, cudaStream_t stream = nullptr);
};
}// namespace luisa::compute::cuda::dcub
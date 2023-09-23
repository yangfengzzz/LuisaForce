//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "runtime/ext/cuda/lcub/dcub_common.h"

namespace luisa::compute::cuda::dcub {

class DCUB_API DeviceScan {
    // DOC:  https://nvlabs.github.io/cub/structcub_1_1_device_scan.html
public:
    static cudaError_t ExclusiveSum(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_in, int32_t *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t ExclusiveSum(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_in, uint32_t *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t ExclusiveSum(void *d_temp_storage, size_t &temp_storage_bytes, const int64_t *d_in, int64_t *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t ExclusiveSum(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_in, uint64_t *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t ExclusiveSum(void *d_temp_storage, size_t &temp_storage_bytes, const float *d_in, float *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t ExclusiveSum(void *d_temp_storage, size_t &temp_storage_bytes, const double *d_in, double *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t InclusiveSum(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_in, int32_t *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t InclusiveSum(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_in, uint32_t *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t InclusiveSum(void *d_temp_storage, size_t &temp_storage_bytes, const int64_t *d_in, int64_t *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t InclusiveSum(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_in, uint64_t *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t InclusiveSum(void *d_temp_storage, size_t &temp_storage_bytes, const float *d_in, float *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t InclusiveSum(void *d_temp_storage, size_t &temp_storage_bytes, const double *d_in, double *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t ExclusiveSumByKey(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, const int32_t *d_values_in, int32_t *d_values_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t ExclusiveSumByKey(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, const uint32_t *d_values_in, uint32_t *d_values_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t ExclusiveSumByKey(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, const int64_t *d_values_in, int64_t *d_values_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t ExclusiveSumByKey(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, const uint64_t *d_values_in, uint64_t *d_values_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t ExclusiveSumByKey(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, const float *d_values_in, float *d_values_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t ExclusiveSumByKey(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, const double *d_values_in, double *d_values_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t InclusiveSumByKey(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, const int32_t *d_values_in, int32_t *d_values_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t InclusiveSumByKey(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, const uint32_t *d_values_in, uint32_t *d_values_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t InclusiveSumByKey(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, const int64_t *d_values_in, int64_t *d_values_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t InclusiveSumByKey(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, const uint64_t *d_values_in, uint64_t *d_values_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t InclusiveSumByKey(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, const float *d_values_in, float *d_values_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t InclusiveSumByKey(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, const double *d_values_in, double *d_values_out, int num_items, cudaStream_t stream = nullptr);
};
}// namespace luisa::compute::cuda::dcub
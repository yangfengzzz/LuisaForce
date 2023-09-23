//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "runtime/ext/cuda/lcub/dcub_common.h"

namespace luisa::compute::cuda::dcub {

class DCUB_API DeviceReduce {
    // DOC:  https://nvlabs.github.io/cub/structcub_1_1_device_reduce.html
public:
    static cudaError_t Sum(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_in, int32_t *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Sum(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_in, uint32_t *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Sum(void *d_temp_storage, size_t &temp_storage_bytes, const int64_t *d_in, int64_t *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Sum(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_in, uint64_t *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Sum(void *d_temp_storage, size_t &temp_storage_bytes, const float *d_in, float *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Sum(void *d_temp_storage, size_t &temp_storage_bytes, const double *d_in, double *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Max(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_in, int32_t *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Max(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_in, uint32_t *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Max(void *d_temp_storage, size_t &temp_storage_bytes, const int64_t *d_in, int64_t *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Max(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_in, uint64_t *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Max(void *d_temp_storage, size_t &temp_storage_bytes, const float *d_in, float *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Max(void *d_temp_storage, size_t &temp_storage_bytes, const double *d_in, double *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Min(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_in, int32_t *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Min(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_in, uint32_t *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Min(void *d_temp_storage, size_t &temp_storage_bytes, const int64_t *d_in, int64_t *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Min(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_in, uint64_t *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Min(void *d_temp_storage, size_t &temp_storage_bytes, const float *d_in, float *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Min(void *d_temp_storage, size_t &temp_storage_bytes, const double *d_in, double *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t ArgMin(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_in, KeyValuePair<int32_t, int32_t> *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t ArgMin(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_in, KeyValuePair<int32_t, uint32_t> *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t ArgMin(void *d_temp_storage, size_t &temp_storage_bytes, const int64_t *d_in, KeyValuePair<int32_t, int64_t> *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t ArgMin(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_in, KeyValuePair<int32_t, uint64_t> *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t ArgMin(void *d_temp_storage, size_t &temp_storage_bytes, const float *d_in, KeyValuePair<int32_t, float> *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t ArgMin(void *d_temp_storage, size_t &temp_storage_bytes, const double *d_in, KeyValuePair<int32_t, double> *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t ArgMax(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_in, KeyValuePair<int32_t, int32_t> *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t ArgMax(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_in, KeyValuePair<int32_t, uint32_t> *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t ArgMax(void *d_temp_storage, size_t &temp_storage_bytes, const int64_t *d_in, KeyValuePair<int32_t, int64_t> *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t ArgMax(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_in, KeyValuePair<int32_t, uint64_t> *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t ArgMax(void *d_temp_storage, size_t &temp_storage_bytes, const float *d_in, KeyValuePair<int32_t, float> *d_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t ArgMax(void *d_temp_storage, size_t &temp_storage_bytes, const double *d_in, KeyValuePair<int32_t, double> *d_out, int num_items, cudaStream_t stream = nullptr);
};
}// namespace luisa::compute::cuda::dcub
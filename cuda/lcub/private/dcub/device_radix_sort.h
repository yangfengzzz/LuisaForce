//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "runtime/ext/cuda/lcub/dcub_common.h"

namespace luisa::compute::cuda::dcub {

class DCUB_API DeviceRadixSort {
    // DOC:  https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html
public:
    static cudaError_t SortPairs(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, int32_t *d_keys_out, const int32_t *d_values_in, int32_t *d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8, cudaStream_t stream = nullptr);

    static cudaError_t SortPairs(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, int32_t *d_keys_out, const uint32_t *d_values_in, uint32_t *d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8, cudaStream_t stream = nullptr);

    static cudaError_t SortPairs(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, int32_t *d_keys_out, const int64_t *d_values_in, int64_t *d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8, cudaStream_t stream = nullptr);

    static cudaError_t SortPairs(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, int32_t *d_keys_out, const uint64_t *d_values_in, uint64_t *d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8, cudaStream_t stream = nullptr);

    static cudaError_t SortPairs(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, int32_t *d_keys_out, const float *d_values_in, float *d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8, cudaStream_t stream = nullptr);

    static cudaError_t SortPairs(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, int32_t *d_keys_out, const double *d_values_in, double *d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8, cudaStream_t stream = nullptr);

    static cudaError_t SortPairsDescending(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, int32_t *d_keys_out, const int32_t *d_values_in, int32_t *d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8, cudaStream_t stream = nullptr);

    static cudaError_t SortPairsDescending(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, int32_t *d_keys_out, const uint32_t *d_values_in, uint32_t *d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8, cudaStream_t stream = nullptr);

    static cudaError_t SortPairsDescending(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, int32_t *d_keys_out, const int64_t *d_values_in, int64_t *d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8, cudaStream_t stream = nullptr);

    static cudaError_t SortPairsDescending(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, int32_t *d_keys_out, const uint64_t *d_values_in, uint64_t *d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8, cudaStream_t stream = nullptr);

    static cudaError_t SortPairsDescending(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, int32_t *d_keys_out, const float *d_values_in, float *d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8, cudaStream_t stream = nullptr);

    static cudaError_t SortPairsDescending(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, int32_t *d_keys_out, const double *d_values_in, double *d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8, cudaStream_t stream = nullptr);

    static cudaError_t SortKeys(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, int32_t *d_keys_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8, cudaStream_t stream = nullptr);

    static cudaError_t SortKeysDescending(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_keys_in, int32_t *d_keys_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8, cudaStream_t stream = nullptr);
};
}// namespace luisa::compute::cuda::dcub
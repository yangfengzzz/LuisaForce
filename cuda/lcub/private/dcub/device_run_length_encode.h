//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "runtime/ext/cuda/lcub/dcub_common.h"

namespace luisa::compute::cuda::dcub {

class DCUB_API DeviceRunLengthEncode {
    // DOC:  https://nvlabs.github.io/cub/structcub_1_1_device_run_length_encode.html
public:
    static cudaError_t Encode(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_in, int32_t *d_unique_out, int32_t *d_counts_out, int32_t *d_num_runs_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Encode(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_in, uint32_t *d_unique_out, int32_t *d_counts_out, int32_t *d_num_runs_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Encode(void *d_temp_storage, size_t &temp_storage_bytes, const int64_t *d_in, int64_t *d_unique_out, int32_t *d_counts_out, int32_t *d_num_runs_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t Encode(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_in, uint64_t *d_unique_out, int32_t *d_counts_out, int32_t *d_num_runs_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t NonTrivialRuns(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_in, int32_t *d_offsets_out, int32_t *d_lengths_out, int32_t *d_num_runs_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t NonTrivialRuns(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_in, int32_t *d_offsets_out, int32_t *d_lengths_out, int32_t *d_num_runs_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t NonTrivialRuns(void *d_temp_storage, size_t &temp_storage_bytes, const int64_t *d_in, int32_t *d_offsets_out, int32_t *d_lengths_out, int32_t *d_num_runs_out, int num_items, cudaStream_t stream = nullptr);

    static cudaError_t NonTrivialRuns(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_in, int32_t *d_offsets_out, int32_t *d_lengths_out, int32_t *d_num_runs_out, int num_items, cudaStream_t stream = nullptr);
};
}// namespace luisa::compute::cuda::dcub
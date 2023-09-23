//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "runtime/ext/cuda/lcub/dcub_common.h"

namespace luisa::compute::cuda::dcub {

class DCUB_API DeviceSpmv {
    // DOC:  https://nvlabs.github.io/cub/structcub_1_1_device_spmv.html
public:
    static cudaError_t CsrMV(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_values, const int *d_row_offsets, const int *d_column_indices, const int32_t *d_vector_x, int32_t *d_vector_y, int num_rows, int num_cols, int num_nonzeros, cudaStream_t stream = nullptr);

    static cudaError_t CsrMV(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_values, const int *d_row_offsets, const int *d_column_indices, const uint32_t *d_vector_x, uint32_t *d_vector_y, int num_rows, int num_cols, int num_nonzeros, cudaStream_t stream = nullptr);

    static cudaError_t CsrMV(void *d_temp_storage, size_t &temp_storage_bytes, const int64_t *d_values, const int *d_row_offsets, const int *d_column_indices, const int64_t *d_vector_x, int64_t *d_vector_y, int num_rows, int num_cols, int num_nonzeros, cudaStream_t stream = nullptr);

    static cudaError_t CsrMV(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_values, const int *d_row_offsets, const int *d_column_indices, const uint64_t *d_vector_x, uint64_t *d_vector_y, int num_rows, int num_cols, int num_nonzeros, cudaStream_t stream = nullptr);

    static cudaError_t CsrMV(void *d_temp_storage, size_t &temp_storage_bytes, const float *d_values, const int *d_row_offsets, const int *d_column_indices, const float *d_vector_x, float *d_vector_y, int num_rows, int num_cols, int num_nonzeros, cudaStream_t stream = nullptr);

    static cudaError_t CsrMV(void *d_temp_storage, size_t &temp_storage_bytes, const double *d_values, const int *d_row_offsets, const int *d_column_indices, const double *d_vector_x, double *d_vector_y, int num_rows, int num_cols, int num_nonzeros, cudaStream_t stream = nullptr);
};
}// namespace luisa::compute::cuda::dcub
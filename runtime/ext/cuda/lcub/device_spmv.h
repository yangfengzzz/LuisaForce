//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/dll_export.h"
#include "runtime/ext/cuda/lcub/dcub_common.h"
#include "runtime/ext/cuda/lcub/cuda_lcub_command.h"

namespace luisa::compute::cuda::lcub {
// DOC:  https://nvlabs.github.io/cub/structcub_1_1_device_spmv.html
class LC_BACKEND_API DeviceSpmv {
    template<typename T>
    using BufferView = luisa::compute::BufferView<T>;
    using UCommand = luisa::unique_ptr<luisa::compute::cuda::CudaLCubCommand>;
public:

    static void CsrMV(size_t &temp_storage_size, BufferView<int32_t> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<int32_t> d_vector_x, BufferView<int32_t> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept;
    static UCommand CsrMV(BufferView<int> d_temp_storage, BufferView<int32_t> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<int32_t> d_vector_x, BufferView<int32_t> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept;

    static void CsrMV(size_t &temp_storage_size, BufferView<uint32_t> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<uint32_t> d_vector_x, BufferView<uint32_t> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept;
    static UCommand CsrMV(BufferView<int> d_temp_storage, BufferView<uint32_t> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<uint32_t> d_vector_x, BufferView<uint32_t> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept;

    static void CsrMV(size_t &temp_storage_size, BufferView<int64_t> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<int64_t> d_vector_x, BufferView<int64_t> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept;
    static UCommand CsrMV(BufferView<int> d_temp_storage, BufferView<int64_t> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<int64_t> d_vector_x, BufferView<int64_t> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept;

    static void CsrMV(size_t &temp_storage_size, BufferView<uint64_t> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<uint64_t> d_vector_x, BufferView<uint64_t> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept;
    static UCommand CsrMV(BufferView<int> d_temp_storage, BufferView<uint64_t> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<uint64_t> d_vector_x, BufferView<uint64_t> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept;

    static void CsrMV(size_t &temp_storage_size, BufferView<float> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<float> d_vector_x, BufferView<float> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept;
    static UCommand CsrMV(BufferView<int> d_temp_storage, BufferView<float> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<float> d_vector_x, BufferView<float> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept;

    static void CsrMV(size_t &temp_storage_size, BufferView<double> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<double> d_vector_x, BufferView<double> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept;
    static UCommand CsrMV(BufferView<int> d_temp_storage, BufferView<double> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<double> d_vector_x, BufferView<double> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept;
};
}// namespace luisa::compute::cuda::lcub
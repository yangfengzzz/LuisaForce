//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/ext/cuda/lcub/device_spmv.h"
#include "private/lcub_utils.h"
#include "private/dcub/device_spmv.h"

namespace luisa::compute::cuda::lcub {
// DOC:  https://nvlabs.github.io/cub/structcub_1_1_device_spmv.html

void DeviceSpmv::CsrMV(size_t &temp_storage_size, BufferView<int32_t> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<int32_t> d_vector_x, BufferView<int32_t> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceSpmv::CsrMV(nullptr, raw(temp_storage_bytes), raw(d_values), raw(d_row_offsets), raw(d_column_indices), raw(d_vector_x), raw(d_vector_y), raw(num_rows), raw(num_cols), raw(num_nonzeros), nullptr);
    });
}

DeviceSpmv::UCommand DeviceSpmv::CsrMV(BufferView<int> d_temp_storage, BufferView<int32_t> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<int32_t> d_vector_x, BufferView<int32_t> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceSpmv::CsrMV(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_values), raw(d_row_offsets), raw(d_column_indices), raw(d_vector_x), raw(d_vector_y), raw(num_rows), raw(num_cols), raw(num_nonzeros), raw(stream));
        });
    });
}

void DeviceSpmv::CsrMV(size_t &temp_storage_size, BufferView<uint32_t> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<uint32_t> d_vector_x, BufferView<uint32_t> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceSpmv::CsrMV(nullptr, raw(temp_storage_bytes), raw(d_values), raw(d_row_offsets), raw(d_column_indices), raw(d_vector_x), raw(d_vector_y), raw(num_rows), raw(num_cols), raw(num_nonzeros), nullptr);
    });
}

DeviceSpmv::UCommand DeviceSpmv::CsrMV(BufferView<int> d_temp_storage, BufferView<uint32_t> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<uint32_t> d_vector_x, BufferView<uint32_t> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceSpmv::CsrMV(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_values), raw(d_row_offsets), raw(d_column_indices), raw(d_vector_x), raw(d_vector_y), raw(num_rows), raw(num_cols), raw(num_nonzeros), raw(stream));
        });
    });
}

void DeviceSpmv::CsrMV(size_t &temp_storage_size, BufferView<int64_t> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<int64_t> d_vector_x, BufferView<int64_t> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceSpmv::CsrMV(nullptr, raw(temp_storage_bytes), raw(d_values), raw(d_row_offsets), raw(d_column_indices), raw(d_vector_x), raw(d_vector_y), raw(num_rows), raw(num_cols), raw(num_nonzeros), nullptr);
    });
}

DeviceSpmv::UCommand DeviceSpmv::CsrMV(BufferView<int> d_temp_storage, BufferView<int64_t> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<int64_t> d_vector_x, BufferView<int64_t> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceSpmv::CsrMV(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_values), raw(d_row_offsets), raw(d_column_indices), raw(d_vector_x), raw(d_vector_y), raw(num_rows), raw(num_cols), raw(num_nonzeros), raw(stream));
        });
    });
}

void DeviceSpmv::CsrMV(size_t &temp_storage_size, BufferView<uint64_t> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<uint64_t> d_vector_x, BufferView<uint64_t> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceSpmv::CsrMV(nullptr, raw(temp_storage_bytes), raw(d_values), raw(d_row_offsets), raw(d_column_indices), raw(d_vector_x), raw(d_vector_y), raw(num_rows), raw(num_cols), raw(num_nonzeros), nullptr);
    });
}

DeviceSpmv::UCommand DeviceSpmv::CsrMV(BufferView<int> d_temp_storage, BufferView<uint64_t> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<uint64_t> d_vector_x, BufferView<uint64_t> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceSpmv::CsrMV(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_values), raw(d_row_offsets), raw(d_column_indices), raw(d_vector_x), raw(d_vector_y), raw(num_rows), raw(num_cols), raw(num_nonzeros), raw(stream));
        });
    });
}

void DeviceSpmv::CsrMV(size_t &temp_storage_size, BufferView<float> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<float> d_vector_x, BufferView<float> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceSpmv::CsrMV(nullptr, raw(temp_storage_bytes), raw(d_values), raw(d_row_offsets), raw(d_column_indices), raw(d_vector_x), raw(d_vector_y), raw(num_rows), raw(num_cols), raw(num_nonzeros), nullptr);
    });
}

DeviceSpmv::UCommand DeviceSpmv::CsrMV(BufferView<int> d_temp_storage, BufferView<float> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<float> d_vector_x, BufferView<float> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceSpmv::CsrMV(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_values), raw(d_row_offsets), raw(d_column_indices), raw(d_vector_x), raw(d_vector_y), raw(num_rows), raw(num_cols), raw(num_nonzeros), raw(stream));
        });
    });
}

void DeviceSpmv::CsrMV(size_t &temp_storage_size, BufferView<double> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<double> d_vector_x, BufferView<double> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceSpmv::CsrMV(nullptr, raw(temp_storage_bytes), raw(d_values), raw(d_row_offsets), raw(d_column_indices), raw(d_vector_x), raw(d_vector_y), raw(num_rows), raw(num_cols), raw(num_nonzeros), nullptr);
    });
}

DeviceSpmv::UCommand DeviceSpmv::CsrMV(BufferView<int> d_temp_storage, BufferView<double> d_values, BufferView<int> d_row_offsets, BufferView<int> d_column_indices, BufferView<double> d_vector_x, BufferView<double> d_vector_y, int num_rows, int num_cols, int num_nonzeros) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceSpmv::CsrMV(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_values), raw(d_row_offsets), raw(d_column_indices), raw(d_vector_x), raw(d_vector_y), raw(num_rows), raw(num_cols), raw(num_nonzeros), raw(stream));
        });
    });
}
}// namespace luisa::compute::cuda::lcub

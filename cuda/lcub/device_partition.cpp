//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/ext/cuda/lcub/device_partition.h"
#include "private/lcub_utils.h"
#include "private/dcub/device_partition.h"

namespace luisa::compute::cuda::lcub {
// DOC:  https://nvlabs.github.io/cub/structcub_1_1_device_partition.html

void DevicePartition::Flagged(size_t &temp_storage_size, BufferView<int32_t> d_in, BufferView<int32_t> d_flags, BufferView<int32_t> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DevicePartition::Flagged(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), nullptr);
    });
}

DevicePartition::UCommand DevicePartition::Flagged(BufferView<int> d_temp_storage, BufferView<int32_t> d_in, BufferView<int32_t> d_flags, BufferView<int32_t> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DevicePartition::Flagged(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), raw(stream));
        });
    });
}

void DevicePartition::Flagged(size_t &temp_storage_size, BufferView<uint32_t> d_in, BufferView<int32_t> d_flags, BufferView<uint32_t> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DevicePartition::Flagged(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), nullptr);
    });
}

DevicePartition::UCommand DevicePartition::Flagged(BufferView<int> d_temp_storage, BufferView<uint32_t> d_in, BufferView<int32_t> d_flags, BufferView<uint32_t> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DevicePartition::Flagged(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), raw(stream));
        });
    });
}

void DevicePartition::Flagged(size_t &temp_storage_size, BufferView<int64_t> d_in, BufferView<int32_t> d_flags, BufferView<int64_t> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DevicePartition::Flagged(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), nullptr);
    });
}

DevicePartition::UCommand DevicePartition::Flagged(BufferView<int> d_temp_storage, BufferView<int64_t> d_in, BufferView<int32_t> d_flags, BufferView<int64_t> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DevicePartition::Flagged(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), raw(stream));
        });
    });
}

void DevicePartition::Flagged(size_t &temp_storage_size, BufferView<uint64_t> d_in, BufferView<int32_t> d_flags, BufferView<uint64_t> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DevicePartition::Flagged(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), nullptr);
    });
}

DevicePartition::UCommand DevicePartition::Flagged(BufferView<int> d_temp_storage, BufferView<uint64_t> d_in, BufferView<int32_t> d_flags, BufferView<uint64_t> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DevicePartition::Flagged(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), raw(stream));
        });
    });
}

void DevicePartition::Flagged(size_t &temp_storage_size, BufferView<float> d_in, BufferView<int32_t> d_flags, BufferView<float> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DevicePartition::Flagged(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), nullptr);
    });
}

DevicePartition::UCommand DevicePartition::Flagged(BufferView<int> d_temp_storage, BufferView<float> d_in, BufferView<int32_t> d_flags, BufferView<float> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DevicePartition::Flagged(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), raw(stream));
        });
    });
}

void DevicePartition::Flagged(size_t &temp_storage_size, BufferView<double> d_in, BufferView<int32_t> d_flags, BufferView<double> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DevicePartition::Flagged(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), nullptr);
    });
}

DevicePartition::UCommand DevicePartition::Flagged(BufferView<int> d_temp_storage, BufferView<double> d_in, BufferView<int32_t> d_flags, BufferView<double> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DevicePartition::Flagged(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), raw(stream));
        });
    });
}
}// namespace luisa::compute::cuda::lcub

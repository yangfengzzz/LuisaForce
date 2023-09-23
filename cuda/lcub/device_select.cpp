//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/ext/cuda/lcub/device_select.h"
#include "private/lcub_utils.h"
#include "private/dcub/device_select.h"

namespace luisa::compute::cuda::lcub {
// DOC:  https://nvlabs.github.io/cub/structcub_1_1_device_select.html

void DeviceSelect::Flagged(size_t &temp_storage_size, BufferView<int32_t> d_in, BufferView<int32_t> d_flags, BufferView<int32_t> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceSelect::Flagged(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), nullptr);
    });
}

DeviceSelect::UCommand DeviceSelect::Flagged(BufferView<int> d_temp_storage, BufferView<int32_t> d_in, BufferView<int32_t> d_flags, BufferView<int32_t> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceSelect::Flagged(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceSelect::Flagged(size_t &temp_storage_size, BufferView<uint32_t> d_in, BufferView<int32_t> d_flags, BufferView<uint32_t> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceSelect::Flagged(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), nullptr);
    });
}

DeviceSelect::UCommand DeviceSelect::Flagged(BufferView<int> d_temp_storage, BufferView<uint32_t> d_in, BufferView<int32_t> d_flags, BufferView<uint32_t> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceSelect::Flagged(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceSelect::Flagged(size_t &temp_storage_size, BufferView<int64_t> d_in, BufferView<int32_t> d_flags, BufferView<int64_t> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceSelect::Flagged(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), nullptr);
    });
}

DeviceSelect::UCommand DeviceSelect::Flagged(BufferView<int> d_temp_storage, BufferView<int64_t> d_in, BufferView<int32_t> d_flags, BufferView<int64_t> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceSelect::Flagged(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceSelect::Flagged(size_t &temp_storage_size, BufferView<uint64_t> d_in, BufferView<int32_t> d_flags, BufferView<uint64_t> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceSelect::Flagged(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), nullptr);
    });
}

DeviceSelect::UCommand DeviceSelect::Flagged(BufferView<int> d_temp_storage, BufferView<uint64_t> d_in, BufferView<int32_t> d_flags, BufferView<uint64_t> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceSelect::Flagged(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceSelect::Flagged(size_t &temp_storage_size, BufferView<float> d_in, BufferView<int32_t> d_flags, BufferView<float> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceSelect::Flagged(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), nullptr);
    });
}

DeviceSelect::UCommand DeviceSelect::Flagged(BufferView<int> d_temp_storage, BufferView<float> d_in, BufferView<int32_t> d_flags, BufferView<float> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceSelect::Flagged(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceSelect::Flagged(size_t &temp_storage_size, BufferView<double> d_in, BufferView<int32_t> d_flags, BufferView<double> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceSelect::Flagged(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), nullptr);
    });
}

DeviceSelect::UCommand DeviceSelect::Flagged(BufferView<int> d_temp_storage, BufferView<double> d_in, BufferView<int32_t> d_flags, BufferView<double> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceSelect::Flagged(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_flags), raw(d_out), raw(d_num_selected_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceSelect::Unique(size_t &temp_storage_size, BufferView<int32_t> d_in, BufferView<int32_t> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceSelect::Unique(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(d_num_selected_out), raw(num_items), nullptr);
    });
}

DeviceSelect::UCommand DeviceSelect::Unique(BufferView<int> d_temp_storage, BufferView<int32_t> d_in, BufferView<int32_t> d_out, BufferView<int32_t> d_num_selected_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceSelect::Unique(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(d_num_selected_out), raw(num_items), raw(stream));
        });
    });
}
}// namespace luisa::compute::cuda::lcub

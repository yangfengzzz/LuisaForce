//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/ext/cuda/lcub/device_radix_sort.h"
#include "private/lcub_utils.h"
#include "private/dcub/device_radix_sort.h"

namespace luisa::compute::cuda::lcub {
// DOC:  https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html

void DeviceRadixSort::SortPairs(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<int32_t> d_values_in, BufferView<int32_t> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceRadixSort::SortPairs(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), nullptr);
    });
}

DeviceRadixSort::UCommand DeviceRadixSort::SortPairs(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<int32_t> d_values_in, BufferView<int32_t> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceRadixSort::SortPairs(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), raw(stream));
        });
    });
}

void DeviceRadixSort::SortPairs(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<uint32_t> d_values_in, BufferView<uint32_t> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceRadixSort::SortPairs(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), nullptr);
    });
}

DeviceRadixSort::UCommand DeviceRadixSort::SortPairs(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<uint32_t> d_values_in, BufferView<uint32_t> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceRadixSort::SortPairs(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), raw(stream));
        });
    });
}

void DeviceRadixSort::SortPairs(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<int64_t> d_values_in, BufferView<int64_t> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceRadixSort::SortPairs(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), nullptr);
    });
}

DeviceRadixSort::UCommand DeviceRadixSort::SortPairs(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<int64_t> d_values_in, BufferView<int64_t> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceRadixSort::SortPairs(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), raw(stream));
        });
    });
}

void DeviceRadixSort::SortPairs(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<uint64_t> d_values_in, BufferView<uint64_t> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceRadixSort::SortPairs(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), nullptr);
    });
}

DeviceRadixSort::UCommand DeviceRadixSort::SortPairs(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<uint64_t> d_values_in, BufferView<uint64_t> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceRadixSort::SortPairs(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), raw(stream));
        });
    });
}

void DeviceRadixSort::SortPairs(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<float> d_values_in, BufferView<float> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceRadixSort::SortPairs(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), nullptr);
    });
}

DeviceRadixSort::UCommand DeviceRadixSort::SortPairs(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<float> d_values_in, BufferView<float> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceRadixSort::SortPairs(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), raw(stream));
        });
    });
}

void DeviceRadixSort::SortPairs(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<double> d_values_in, BufferView<double> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceRadixSort::SortPairs(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), nullptr);
    });
}

DeviceRadixSort::UCommand DeviceRadixSort::SortPairs(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<double> d_values_in, BufferView<double> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceRadixSort::SortPairs(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), raw(stream));
        });
    });
}

void DeviceRadixSort::SortPairsDescending(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<int32_t> d_values_in, BufferView<int32_t> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceRadixSort::SortPairsDescending(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), nullptr);
    });
}

DeviceRadixSort::UCommand DeviceRadixSort::SortPairsDescending(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<int32_t> d_values_in, BufferView<int32_t> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceRadixSort::SortPairsDescending(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), raw(stream));
        });
    });
}

void DeviceRadixSort::SortPairsDescending(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<uint32_t> d_values_in, BufferView<uint32_t> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceRadixSort::SortPairsDescending(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), nullptr);
    });
}

DeviceRadixSort::UCommand DeviceRadixSort::SortPairsDescending(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<uint32_t> d_values_in, BufferView<uint32_t> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceRadixSort::SortPairsDescending(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), raw(stream));
        });
    });
}

void DeviceRadixSort::SortPairsDescending(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<int64_t> d_values_in, BufferView<int64_t> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceRadixSort::SortPairsDescending(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), nullptr);
    });
}

DeviceRadixSort::UCommand DeviceRadixSort::SortPairsDescending(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<int64_t> d_values_in, BufferView<int64_t> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceRadixSort::SortPairsDescending(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), raw(stream));
        });
    });
}

void DeviceRadixSort::SortPairsDescending(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<uint64_t> d_values_in, BufferView<uint64_t> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceRadixSort::SortPairsDescending(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), nullptr);
    });
}

DeviceRadixSort::UCommand DeviceRadixSort::SortPairsDescending(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<uint64_t> d_values_in, BufferView<uint64_t> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceRadixSort::SortPairsDescending(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), raw(stream));
        });
    });
}

void DeviceRadixSort::SortPairsDescending(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<float> d_values_in, BufferView<float> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceRadixSort::SortPairsDescending(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), nullptr);
    });
}

DeviceRadixSort::UCommand DeviceRadixSort::SortPairsDescending(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<float> d_values_in, BufferView<float> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceRadixSort::SortPairsDescending(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), raw(stream));
        });
    });
}

void DeviceRadixSort::SortPairsDescending(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<double> d_values_in, BufferView<double> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceRadixSort::SortPairsDescending(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), nullptr);
    });
}

DeviceRadixSort::UCommand DeviceRadixSort::SortPairsDescending(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<double> d_values_in, BufferView<double> d_values_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceRadixSort::SortPairsDescending(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(d_values_in), raw(d_values_out), raw(num_items), raw(begin_bit), raw(end_bit), raw(stream));
        });
    });
}

void DeviceRadixSort::SortKeys(size_t &temp_storage_size, const int32_t *d_keys_in, int32_t *d_keys_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceRadixSort::SortKeys(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(num_items), raw(begin_bit), raw(end_bit), nullptr);
    });
}

DeviceRadixSort::UCommand DeviceRadixSort::SortKeys(BufferView<int> d_temp_storage, const int32_t *d_keys_in, int32_t *d_keys_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceRadixSort::SortKeys(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(num_items), raw(begin_bit), raw(end_bit), raw(stream));
        });
    });
}

void DeviceRadixSort::SortKeysDescending(size_t &temp_storage_size, const int32_t *d_keys_in, int32_t *d_keys_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceRadixSort::SortKeysDescending(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(num_items), raw(begin_bit), raw(end_bit), nullptr);
    });
}

DeviceRadixSort::UCommand DeviceRadixSort::SortKeysDescending(BufferView<int> d_temp_storage, const int32_t *d_keys_in, int32_t *d_keys_out, int num_items, int begin_bit, int end_bit) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceRadixSort::SortKeysDescending(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_keys_out), raw(num_items), raw(begin_bit), raw(end_bit), raw(stream));
        });
    });
}
}// namespace luisa::compute::cuda::lcub

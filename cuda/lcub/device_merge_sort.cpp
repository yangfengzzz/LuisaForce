//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/ext/cuda/lcub/device_merge_sort.h"
#include "private/lcub_utils.h"
#include "private/dcub/device_merge_sort.h"

namespace luisa::compute::cuda::lcub {
// DOC:  https://nvlabs.github.io/cub/structcub_1_1_device_merge_sort.html

void DeviceMergeSort::SortPairs(size_t &temp_storage_size, BufferView<int32_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortPairs(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortPairs(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortPairs(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::SortPairs(size_t &temp_storage_size, BufferView<uint32_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortPairs(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortPairs(BufferView<int> d_temp_storage, BufferView<uint32_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortPairs(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::SortPairs(size_t &temp_storage_size, BufferView<int64_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortPairs(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortPairs(BufferView<int> d_temp_storage, BufferView<int64_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortPairs(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::SortPairs(size_t &temp_storage_size, BufferView<uint64_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortPairs(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortPairs(BufferView<int> d_temp_storage, BufferView<uint64_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortPairs(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::SortPairs(size_t &temp_storage_size, BufferView<float> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortPairs(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortPairs(BufferView<int> d_temp_storage, BufferView<float> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortPairs(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::SortPairs(size_t &temp_storage_size, BufferView<double> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortPairs(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortPairs(BufferView<int> d_temp_storage, BufferView<double> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortPairs(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::SortPairsCopy(size_t &temp_storage_size, BufferView<int32_t> d_input_keys, BufferView<int32_t> d_input_items, BufferView<int32_t> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortPairsCopy(nullptr, raw(temp_storage_bytes), raw(d_input_keys), raw(d_input_items), raw(d_output_keys), raw(d_output_items), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortPairsCopy(BufferView<int> d_temp_storage, BufferView<int32_t> d_input_keys, BufferView<int32_t> d_input_items, BufferView<int32_t> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortPairsCopy(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_input_keys), raw(d_input_items), raw(d_output_keys), raw(d_output_items), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::SortPairsCopy(size_t &temp_storage_size, BufferView<uint32_t> d_input_keys, BufferView<int32_t> d_input_items, BufferView<uint32_t> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortPairsCopy(nullptr, raw(temp_storage_bytes), raw(d_input_keys), raw(d_input_items), raw(d_output_keys), raw(d_output_items), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortPairsCopy(BufferView<int> d_temp_storage, BufferView<uint32_t> d_input_keys, BufferView<int32_t> d_input_items, BufferView<uint32_t> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortPairsCopy(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_input_keys), raw(d_input_items), raw(d_output_keys), raw(d_output_items), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::SortPairsCopy(size_t &temp_storage_size, BufferView<int64_t> d_input_keys, BufferView<int32_t> d_input_items, BufferView<int64_t> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortPairsCopy(nullptr, raw(temp_storage_bytes), raw(d_input_keys), raw(d_input_items), raw(d_output_keys), raw(d_output_items), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortPairsCopy(BufferView<int> d_temp_storage, BufferView<int64_t> d_input_keys, BufferView<int32_t> d_input_items, BufferView<int64_t> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortPairsCopy(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_input_keys), raw(d_input_items), raw(d_output_keys), raw(d_output_items), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::SortPairsCopy(size_t &temp_storage_size, BufferView<uint64_t> d_input_keys, BufferView<int32_t> d_input_items, BufferView<uint64_t> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortPairsCopy(nullptr, raw(temp_storage_bytes), raw(d_input_keys), raw(d_input_items), raw(d_output_keys), raw(d_output_items), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortPairsCopy(BufferView<int> d_temp_storage, BufferView<uint64_t> d_input_keys, BufferView<int32_t> d_input_items, BufferView<uint64_t> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortPairsCopy(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_input_keys), raw(d_input_items), raw(d_output_keys), raw(d_output_items), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::SortPairsCopy(size_t &temp_storage_size, BufferView<float> d_input_keys, BufferView<int32_t> d_input_items, BufferView<float> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortPairsCopy(nullptr, raw(temp_storage_bytes), raw(d_input_keys), raw(d_input_items), raw(d_output_keys), raw(d_output_items), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortPairsCopy(BufferView<int> d_temp_storage, BufferView<float> d_input_keys, BufferView<int32_t> d_input_items, BufferView<float> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortPairsCopy(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_input_keys), raw(d_input_items), raw(d_output_keys), raw(d_output_items), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::SortPairsCopy(size_t &temp_storage_size, BufferView<double> d_input_keys, BufferView<int32_t> d_input_items, BufferView<double> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortPairsCopy(nullptr, raw(temp_storage_bytes), raw(d_input_keys), raw(d_input_items), raw(d_output_keys), raw(d_output_items), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortPairsCopy(BufferView<int> d_temp_storage, BufferView<double> d_input_keys, BufferView<int32_t> d_input_items, BufferView<double> d_output_keys, BufferView<int32_t> d_output_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortPairsCopy(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_input_keys), raw(d_input_items), raw(d_output_keys), raw(d_output_items), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::SortKeys(size_t &temp_storage_size, BufferView<int32_t> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortKeys(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortKeys(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortKeys(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::SortKeys(size_t &temp_storage_size, BufferView<uint32_t> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortKeys(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortKeys(BufferView<int> d_temp_storage, BufferView<uint32_t> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortKeys(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::SortKeys(size_t &temp_storage_size, BufferView<int64_t> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortKeys(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortKeys(BufferView<int> d_temp_storage, BufferView<int64_t> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortKeys(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::SortKeys(size_t &temp_storage_size, BufferView<uint64_t> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortKeys(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortKeys(BufferView<int> d_temp_storage, BufferView<uint64_t> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortKeys(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::SortKeys(size_t &temp_storage_size, BufferView<float> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortKeys(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortKeys(BufferView<int> d_temp_storage, BufferView<float> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortKeys(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::SortKeys(size_t &temp_storage_size, BufferView<double> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortKeys(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortKeys(BufferView<int> d_temp_storage, BufferView<double> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortKeys(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::SortKeysCopy(size_t &temp_storage_size, BufferView<int32_t> d_input_keys, BufferView<int32_t> d_output_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortKeysCopy(nullptr, raw(temp_storage_bytes), raw(d_input_keys), raw(d_output_keys), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortKeysCopy(BufferView<int> d_temp_storage, BufferView<int32_t> d_input_keys, BufferView<int32_t> d_output_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortKeysCopy(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_input_keys), raw(d_output_keys), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::SortKeysCopy(size_t &temp_storage_size, BufferView<uint32_t> d_input_keys, BufferView<uint32_t> d_output_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortKeysCopy(nullptr, raw(temp_storage_bytes), raw(d_input_keys), raw(d_output_keys), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortKeysCopy(BufferView<int> d_temp_storage, BufferView<uint32_t> d_input_keys, BufferView<uint32_t> d_output_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortKeysCopy(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_input_keys), raw(d_output_keys), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::SortKeysCopy(size_t &temp_storage_size, BufferView<int64_t> d_input_keys, BufferView<int64_t> d_output_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortKeysCopy(nullptr, raw(temp_storage_bytes), raw(d_input_keys), raw(d_output_keys), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortKeysCopy(BufferView<int> d_temp_storage, BufferView<int64_t> d_input_keys, BufferView<int64_t> d_output_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortKeysCopy(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_input_keys), raw(d_output_keys), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::SortKeysCopy(size_t &temp_storage_size, BufferView<uint64_t> d_input_keys, BufferView<uint64_t> d_output_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortKeysCopy(nullptr, raw(temp_storage_bytes), raw(d_input_keys), raw(d_output_keys), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortKeysCopy(BufferView<int> d_temp_storage, BufferView<uint64_t> d_input_keys, BufferView<uint64_t> d_output_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortKeysCopy(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_input_keys), raw(d_output_keys), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::SortKeysCopy(size_t &temp_storage_size, BufferView<float> d_input_keys, BufferView<float> d_output_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortKeysCopy(nullptr, raw(temp_storage_bytes), raw(d_input_keys), raw(d_output_keys), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortKeysCopy(BufferView<int> d_temp_storage, BufferView<float> d_input_keys, BufferView<float> d_output_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortKeysCopy(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_input_keys), raw(d_output_keys), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::SortKeysCopy(size_t &temp_storage_size, BufferView<double> d_input_keys, BufferView<double> d_output_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::SortKeysCopy(nullptr, raw(temp_storage_bytes), raw(d_input_keys), raw(d_output_keys), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::SortKeysCopy(BufferView<int> d_temp_storage, BufferView<double> d_input_keys, BufferView<double> d_output_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::SortKeysCopy(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_input_keys), raw(d_output_keys), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::StableSortPairs(size_t &temp_storage_size, BufferView<int32_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::StableSortPairs(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::StableSortPairs(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::StableSortPairs(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::StableSortPairs(size_t &temp_storage_size, BufferView<uint32_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::StableSortPairs(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::StableSortPairs(BufferView<int> d_temp_storage, BufferView<uint32_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::StableSortPairs(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::StableSortPairs(size_t &temp_storage_size, BufferView<int64_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::StableSortPairs(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::StableSortPairs(BufferView<int> d_temp_storage, BufferView<int64_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::StableSortPairs(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::StableSortPairs(size_t &temp_storage_size, BufferView<uint64_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::StableSortPairs(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::StableSortPairs(BufferView<int> d_temp_storage, BufferView<uint64_t> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::StableSortPairs(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::StableSortPairs(size_t &temp_storage_size, BufferView<float> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::StableSortPairs(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::StableSortPairs(BufferView<int> d_temp_storage, BufferView<float> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::StableSortPairs(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::StableSortPairs(size_t &temp_storage_size, BufferView<double> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::StableSortPairs(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::StableSortPairs(BufferView<int> d_temp_storage, BufferView<double> d_keys, BufferView<int32_t> d_items, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::StableSortPairs(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(d_items), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::StableSortKeys(size_t &temp_storage_size, BufferView<int32_t> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::StableSortKeys(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::StableSortKeys(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::StableSortKeys(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::StableSortKeys(size_t &temp_storage_size, BufferView<uint32_t> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::StableSortKeys(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::StableSortKeys(BufferView<int> d_temp_storage, BufferView<uint32_t> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::StableSortKeys(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::StableSortKeys(size_t &temp_storage_size, BufferView<int64_t> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::StableSortKeys(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::StableSortKeys(BufferView<int> d_temp_storage, BufferView<int64_t> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::StableSortKeys(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::StableSortKeys(size_t &temp_storage_size, BufferView<uint64_t> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::StableSortKeys(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::StableSortKeys(BufferView<int> d_temp_storage, BufferView<uint64_t> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::StableSortKeys(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::StableSortKeys(size_t &temp_storage_size, BufferView<float> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::StableSortKeys(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::StableSortKeys(BufferView<int> d_temp_storage, BufferView<float> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::StableSortKeys(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}

void DeviceMergeSort::StableSortKeys(size_t &temp_storage_size, BufferView<double> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceMergeSort::StableSortKeys(nullptr, raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), nullptr);
    });
}

DeviceMergeSort::UCommand DeviceMergeSort::StableSortKeys(BufferView<int> d_temp_storage, BufferView<double> d_keys, int num_items, dcub::BinaryOperator compare_op) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceMergeSort::StableSortKeys(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys), raw(num_items), raw(compare_op), raw(stream));
        });
    });
}
}// namespace luisa::compute::cuda::lcub

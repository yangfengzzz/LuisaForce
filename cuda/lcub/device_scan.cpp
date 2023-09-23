//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/ext/cuda/lcub/device_scan.h"
#include "private/lcub_utils.h"
#include "private/dcub/device_scan.h"

namespace luisa::compute::cuda::lcub {
// DOC:  https://nvlabs.github.io/cub/structcub_1_1_device_scan.html

void DeviceScan::ExclusiveSum(size_t &temp_storage_size, BufferView<int32_t> d_in, BufferView<int32_t> d_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::ExclusiveSum(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::ExclusiveSum(BufferView<int> d_temp_storage, BufferView<int32_t> d_in, BufferView<int32_t> d_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::ExclusiveSum(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceScan::ExclusiveSum(size_t &temp_storage_size, BufferView<uint32_t> d_in, BufferView<uint32_t> d_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::ExclusiveSum(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::ExclusiveSum(BufferView<int> d_temp_storage, BufferView<uint32_t> d_in, BufferView<uint32_t> d_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::ExclusiveSum(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceScan::ExclusiveSum(size_t &temp_storage_size, BufferView<int64_t> d_in, BufferView<int64_t> d_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::ExclusiveSum(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::ExclusiveSum(BufferView<int> d_temp_storage, BufferView<int64_t> d_in, BufferView<int64_t> d_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::ExclusiveSum(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceScan::ExclusiveSum(size_t &temp_storage_size, BufferView<uint64_t> d_in, BufferView<uint64_t> d_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::ExclusiveSum(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::ExclusiveSum(BufferView<int> d_temp_storage, BufferView<uint64_t> d_in, BufferView<uint64_t> d_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::ExclusiveSum(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceScan::ExclusiveSum(size_t &temp_storage_size, BufferView<float> d_in, BufferView<float> d_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::ExclusiveSum(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::ExclusiveSum(BufferView<int> d_temp_storage, BufferView<float> d_in, BufferView<float> d_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::ExclusiveSum(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceScan::ExclusiveSum(size_t &temp_storage_size, BufferView<double> d_in, BufferView<double> d_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::ExclusiveSum(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::ExclusiveSum(BufferView<int> d_temp_storage, BufferView<double> d_in, BufferView<double> d_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::ExclusiveSum(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceScan::InclusiveSum(size_t &temp_storage_size, BufferView<int32_t> d_in, BufferView<int32_t> d_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::InclusiveSum(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::InclusiveSum(BufferView<int> d_temp_storage, BufferView<int32_t> d_in, BufferView<int32_t> d_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::InclusiveSum(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceScan::InclusiveSum(size_t &temp_storage_size, BufferView<uint32_t> d_in, BufferView<uint32_t> d_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::InclusiveSum(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::InclusiveSum(BufferView<int> d_temp_storage, BufferView<uint32_t> d_in, BufferView<uint32_t> d_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::InclusiveSum(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceScan::InclusiveSum(size_t &temp_storage_size, BufferView<int64_t> d_in, BufferView<int64_t> d_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::InclusiveSum(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::InclusiveSum(BufferView<int> d_temp_storage, BufferView<int64_t> d_in, BufferView<int64_t> d_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::InclusiveSum(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceScan::InclusiveSum(size_t &temp_storage_size, BufferView<uint64_t> d_in, BufferView<uint64_t> d_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::InclusiveSum(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::InclusiveSum(BufferView<int> d_temp_storage, BufferView<uint64_t> d_in, BufferView<uint64_t> d_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::InclusiveSum(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceScan::InclusiveSum(size_t &temp_storage_size, BufferView<float> d_in, BufferView<float> d_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::InclusiveSum(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::InclusiveSum(BufferView<int> d_temp_storage, BufferView<float> d_in, BufferView<float> d_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::InclusiveSum(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceScan::InclusiveSum(size_t &temp_storage_size, BufferView<double> d_in, BufferView<double> d_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::InclusiveSum(nullptr, raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::InclusiveSum(BufferView<int> d_temp_storage, BufferView<double> d_in, BufferView<double> d_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::InclusiveSum(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_in), raw(d_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceScan::ExclusiveSumByKey(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_values_in, BufferView<int32_t> d_values_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::ExclusiveSumByKey(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::ExclusiveSumByKey(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_values_in, BufferView<int32_t> d_values_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::ExclusiveSumByKey(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceScan::ExclusiveSumByKey(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<uint32_t> d_values_in, BufferView<uint32_t> d_values_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::ExclusiveSumByKey(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::ExclusiveSumByKey(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<uint32_t> d_values_in, BufferView<uint32_t> d_values_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::ExclusiveSumByKey(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceScan::ExclusiveSumByKey(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int64_t> d_values_in, BufferView<int64_t> d_values_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::ExclusiveSumByKey(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::ExclusiveSumByKey(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int64_t> d_values_in, BufferView<int64_t> d_values_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::ExclusiveSumByKey(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceScan::ExclusiveSumByKey(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<uint64_t> d_values_in, BufferView<uint64_t> d_values_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::ExclusiveSumByKey(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::ExclusiveSumByKey(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<uint64_t> d_values_in, BufferView<uint64_t> d_values_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::ExclusiveSumByKey(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceScan::ExclusiveSumByKey(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<float> d_values_in, BufferView<float> d_values_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::ExclusiveSumByKey(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::ExclusiveSumByKey(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<float> d_values_in, BufferView<float> d_values_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::ExclusiveSumByKey(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceScan::ExclusiveSumByKey(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<double> d_values_in, BufferView<double> d_values_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::ExclusiveSumByKey(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::ExclusiveSumByKey(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<double> d_values_in, BufferView<double> d_values_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::ExclusiveSumByKey(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceScan::InclusiveSumByKey(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_values_in, BufferView<int32_t> d_values_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::InclusiveSumByKey(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::InclusiveSumByKey(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_values_in, BufferView<int32_t> d_values_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::InclusiveSumByKey(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceScan::InclusiveSumByKey(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<uint32_t> d_values_in, BufferView<uint32_t> d_values_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::InclusiveSumByKey(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::InclusiveSumByKey(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<uint32_t> d_values_in, BufferView<uint32_t> d_values_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::InclusiveSumByKey(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceScan::InclusiveSumByKey(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int64_t> d_values_in, BufferView<int64_t> d_values_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::InclusiveSumByKey(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::InclusiveSumByKey(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int64_t> d_values_in, BufferView<int64_t> d_values_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::InclusiveSumByKey(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceScan::InclusiveSumByKey(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<uint64_t> d_values_in, BufferView<uint64_t> d_values_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::InclusiveSumByKey(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::InclusiveSumByKey(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<uint64_t> d_values_in, BufferView<uint64_t> d_values_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::InclusiveSumByKey(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceScan::InclusiveSumByKey(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<float> d_values_in, BufferView<float> d_values_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::InclusiveSumByKey(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::InclusiveSumByKey(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<float> d_values_in, BufferView<float> d_values_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::InclusiveSumByKey(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), raw(stream));
        });
    });
}

void DeviceScan::InclusiveSumByKey(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<double> d_values_in, BufferView<double> d_values_out, int num_items) noexcept {
    using namespace details;
    inner(temp_storage_size, [&](size_t &temp_storage_bytes) {
        return dcub::DeviceScan::InclusiveSumByKey(nullptr, raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), nullptr);
    });
}

DeviceScan::UCommand DeviceScan::InclusiveSumByKey(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<double> d_values_in, BufferView<double> d_values_out, int num_items) noexcept {
    using namespace details;
    return luisa::make_unique<luisa::compute::cuda::CudaLCubCommand>([=](cudaStream_t stream) {
        inner(d_temp_storage, [&](size_t &temp_storage_bytes) {
            return dcub::DeviceScan::InclusiveSumByKey(raw(d_temp_storage), raw(temp_storage_bytes), raw(d_keys_in), raw(d_values_in), raw(d_values_out), raw(num_items), raw(stream));
        });
    });
}
}// namespace luisa::compute::cuda::lcub

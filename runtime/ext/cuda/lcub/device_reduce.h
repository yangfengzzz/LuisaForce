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
// DOC:  https://nvlabs.github.io/cub/structcub_1_1_device_reduce.html
class LC_BACKEND_API DeviceReduce {
    template<typename T>
    using BufferView = luisa::compute::BufferView<T>;
    using UCommand = luisa::unique_ptr<luisa::compute::cuda::CudaLCubCommand>;
public:

    static void Sum(size_t &temp_storage_size, BufferView<int32_t> d_in, BufferView<int32_t> d_out, int num_items) noexcept;
    static UCommand Sum(BufferView<int> d_temp_storage, BufferView<int32_t> d_in, BufferView<int32_t> d_out, int num_items) noexcept;

    static void Sum(size_t &temp_storage_size, BufferView<uint32_t> d_in, BufferView<uint32_t> d_out, int num_items) noexcept;
    static UCommand Sum(BufferView<int> d_temp_storage, BufferView<uint32_t> d_in, BufferView<uint32_t> d_out, int num_items) noexcept;

    static void Sum(size_t &temp_storage_size, BufferView<int64_t> d_in, BufferView<int64_t> d_out, int num_items) noexcept;
    static UCommand Sum(BufferView<int> d_temp_storage, BufferView<int64_t> d_in, BufferView<int64_t> d_out, int num_items) noexcept;

    static void Sum(size_t &temp_storage_size, BufferView<uint64_t> d_in, BufferView<uint64_t> d_out, int num_items) noexcept;
    static UCommand Sum(BufferView<int> d_temp_storage, BufferView<uint64_t> d_in, BufferView<uint64_t> d_out, int num_items) noexcept;

    static void Sum(size_t &temp_storage_size, BufferView<float> d_in, BufferView<float> d_out, int num_items) noexcept;
    static UCommand Sum(BufferView<int> d_temp_storage, BufferView<float> d_in, BufferView<float> d_out, int num_items) noexcept;

    static void Sum(size_t &temp_storage_size, BufferView<double> d_in, BufferView<double> d_out, int num_items) noexcept;
    static UCommand Sum(BufferView<int> d_temp_storage, BufferView<double> d_in, BufferView<double> d_out, int num_items) noexcept;

    static void Max(size_t &temp_storage_size, BufferView<int32_t> d_in, BufferView<int32_t> d_out, int num_items) noexcept;
    static UCommand Max(BufferView<int> d_temp_storage, BufferView<int32_t> d_in, BufferView<int32_t> d_out, int num_items) noexcept;

    static void Max(size_t &temp_storage_size, BufferView<uint32_t> d_in, BufferView<uint32_t> d_out, int num_items) noexcept;
    static UCommand Max(BufferView<int> d_temp_storage, BufferView<uint32_t> d_in, BufferView<uint32_t> d_out, int num_items) noexcept;

    static void Max(size_t &temp_storage_size, BufferView<int64_t> d_in, BufferView<int64_t> d_out, int num_items) noexcept;
    static UCommand Max(BufferView<int> d_temp_storage, BufferView<int64_t> d_in, BufferView<int64_t> d_out, int num_items) noexcept;

    static void Max(size_t &temp_storage_size, BufferView<uint64_t> d_in, BufferView<uint64_t> d_out, int num_items) noexcept;
    static UCommand Max(BufferView<int> d_temp_storage, BufferView<uint64_t> d_in, BufferView<uint64_t> d_out, int num_items) noexcept;

    static void Max(size_t &temp_storage_size, BufferView<float> d_in, BufferView<float> d_out, int num_items) noexcept;
    static UCommand Max(BufferView<int> d_temp_storage, BufferView<float> d_in, BufferView<float> d_out, int num_items) noexcept;

    static void Max(size_t &temp_storage_size, BufferView<double> d_in, BufferView<double> d_out, int num_items) noexcept;
    static UCommand Max(BufferView<int> d_temp_storage, BufferView<double> d_in, BufferView<double> d_out, int num_items) noexcept;

    static void Min(size_t &temp_storage_size, BufferView<int32_t> d_in, BufferView<int32_t> d_out, int num_items) noexcept;
    static UCommand Min(BufferView<int> d_temp_storage, BufferView<int32_t> d_in, BufferView<int32_t> d_out, int num_items) noexcept;

    static void Min(size_t &temp_storage_size, BufferView<uint32_t> d_in, BufferView<uint32_t> d_out, int num_items) noexcept;
    static UCommand Min(BufferView<int> d_temp_storage, BufferView<uint32_t> d_in, BufferView<uint32_t> d_out, int num_items) noexcept;

    static void Min(size_t &temp_storage_size, BufferView<int64_t> d_in, BufferView<int64_t> d_out, int num_items) noexcept;
    static UCommand Min(BufferView<int> d_temp_storage, BufferView<int64_t> d_in, BufferView<int64_t> d_out, int num_items) noexcept;

    static void Min(size_t &temp_storage_size, BufferView<uint64_t> d_in, BufferView<uint64_t> d_out, int num_items) noexcept;
    static UCommand Min(BufferView<int> d_temp_storage, BufferView<uint64_t> d_in, BufferView<uint64_t> d_out, int num_items) noexcept;

    static void Min(size_t &temp_storage_size, BufferView<float> d_in, BufferView<float> d_out, int num_items) noexcept;
    static UCommand Min(BufferView<int> d_temp_storage, BufferView<float> d_in, BufferView<float> d_out, int num_items) noexcept;

    static void Min(size_t &temp_storage_size, BufferView<double> d_in, BufferView<double> d_out, int num_items) noexcept;
    static UCommand Min(BufferView<int> d_temp_storage, BufferView<double> d_in, BufferView<double> d_out, int num_items) noexcept;

    static void ArgMin(size_t &temp_storage_size, BufferView<int32_t> d_in, BufferView<dcub::KeyValuePair<int32_t, int32_t>> d_out, int num_items) noexcept;
    static UCommand ArgMin(BufferView<int> d_temp_storage, BufferView<int32_t> d_in, BufferView<dcub::KeyValuePair<int32_t, int32_t>> d_out, int num_items) noexcept;

    static void ArgMin(size_t &temp_storage_size, BufferView<uint32_t> d_in, BufferView<dcub::KeyValuePair<int32_t, uint32_t>> d_out, int num_items) noexcept;
    static UCommand ArgMin(BufferView<int> d_temp_storage, BufferView<uint32_t> d_in, BufferView<dcub::KeyValuePair<int32_t, uint32_t>> d_out, int num_items) noexcept;

    static void ArgMin(size_t &temp_storage_size, BufferView<int64_t> d_in, BufferView<dcub::KeyValuePair<int32_t, int64_t>> d_out, int num_items) noexcept;
    static UCommand ArgMin(BufferView<int> d_temp_storage, BufferView<int64_t> d_in, BufferView<dcub::KeyValuePair<int32_t, int64_t>> d_out, int num_items) noexcept;

    static void ArgMin(size_t &temp_storage_size, BufferView<uint64_t> d_in, BufferView<dcub::KeyValuePair<int32_t, uint64_t>> d_out, int num_items) noexcept;
    static UCommand ArgMin(BufferView<int> d_temp_storage, BufferView<uint64_t> d_in, BufferView<dcub::KeyValuePair<int32_t, uint64_t>> d_out, int num_items) noexcept;

    static void ArgMin(size_t &temp_storage_size, BufferView<float> d_in, BufferView<dcub::KeyValuePair<int32_t, float>> d_out, int num_items) noexcept;
    static UCommand ArgMin(BufferView<int> d_temp_storage, BufferView<float> d_in, BufferView<dcub::KeyValuePair<int32_t, float>> d_out, int num_items) noexcept;

    static void ArgMin(size_t &temp_storage_size, BufferView<double> d_in, BufferView<dcub::KeyValuePair<int32_t, double>> d_out, int num_items) noexcept;
    static UCommand ArgMin(BufferView<int> d_temp_storage, BufferView<double> d_in, BufferView<dcub::KeyValuePair<int32_t, double>> d_out, int num_items) noexcept;

    static void ArgMax(size_t &temp_storage_size, BufferView<int32_t> d_in, BufferView<dcub::KeyValuePair<int32_t, int32_t>> d_out, int num_items) noexcept;
    static UCommand ArgMax(BufferView<int> d_temp_storage, BufferView<int32_t> d_in, BufferView<dcub::KeyValuePair<int32_t, int32_t>> d_out, int num_items) noexcept;

    static void ArgMax(size_t &temp_storage_size, BufferView<uint32_t> d_in, BufferView<dcub::KeyValuePair<int32_t, uint32_t>> d_out, int num_items) noexcept;
    static UCommand ArgMax(BufferView<int> d_temp_storage, BufferView<uint32_t> d_in, BufferView<dcub::KeyValuePair<int32_t, uint32_t>> d_out, int num_items) noexcept;

    static void ArgMax(size_t &temp_storage_size, BufferView<int64_t> d_in, BufferView<dcub::KeyValuePair<int32_t, int64_t>> d_out, int num_items) noexcept;
    static UCommand ArgMax(BufferView<int> d_temp_storage, BufferView<int64_t> d_in, BufferView<dcub::KeyValuePair<int32_t, int64_t>> d_out, int num_items) noexcept;

    static void ArgMax(size_t &temp_storage_size, BufferView<uint64_t> d_in, BufferView<dcub::KeyValuePair<int32_t, uint64_t>> d_out, int num_items) noexcept;
    static UCommand ArgMax(BufferView<int> d_temp_storage, BufferView<uint64_t> d_in, BufferView<dcub::KeyValuePair<int32_t, uint64_t>> d_out, int num_items) noexcept;

    static void ArgMax(size_t &temp_storage_size, BufferView<float> d_in, BufferView<dcub::KeyValuePair<int32_t, float>> d_out, int num_items) noexcept;
    static UCommand ArgMax(BufferView<int> d_temp_storage, BufferView<float> d_in, BufferView<dcub::KeyValuePair<int32_t, float>> d_out, int num_items) noexcept;

    static void ArgMax(size_t &temp_storage_size, BufferView<double> d_in, BufferView<dcub::KeyValuePair<int32_t, double>> d_out, int num_items) noexcept;
    static UCommand ArgMax(BufferView<int> d_temp_storage, BufferView<double> d_in, BufferView<dcub::KeyValuePair<int32_t, double>> d_out, int num_items) noexcept;
};
}// namespace luisa::compute::cuda::lcub
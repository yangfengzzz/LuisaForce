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
// DOC:  https://nvlabs.github.io/cub/structcub_1_1_device_radix_sort.html
class LC_BACKEND_API DeviceRadixSort {
    template<typename T>
    using BufferView = luisa::compute::BufferView<T>;
    using UCommand = luisa::unique_ptr<luisa::compute::cuda::CudaLCubCommand>;
public:

    static void SortPairs(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<int32_t> d_values_in, BufferView<int32_t> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;
    static UCommand SortPairs(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<int32_t> d_values_in, BufferView<int32_t> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;

    static void SortPairs(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<uint32_t> d_values_in, BufferView<uint32_t> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;
    static UCommand SortPairs(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<uint32_t> d_values_in, BufferView<uint32_t> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;

    static void SortPairs(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<int64_t> d_values_in, BufferView<int64_t> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;
    static UCommand SortPairs(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<int64_t> d_values_in, BufferView<int64_t> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;

    static void SortPairs(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<uint64_t> d_values_in, BufferView<uint64_t> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;
    static UCommand SortPairs(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<uint64_t> d_values_in, BufferView<uint64_t> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;

    static void SortPairs(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<float> d_values_in, BufferView<float> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;
    static UCommand SortPairs(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<float> d_values_in, BufferView<float> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;

    static void SortPairs(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<double> d_values_in, BufferView<double> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;
    static UCommand SortPairs(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<double> d_values_in, BufferView<double> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;

    static void SortPairsDescending(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<int32_t> d_values_in, BufferView<int32_t> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;
    static UCommand SortPairsDescending(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<int32_t> d_values_in, BufferView<int32_t> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;

    static void SortPairsDescending(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<uint32_t> d_values_in, BufferView<uint32_t> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;
    static UCommand SortPairsDescending(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<uint32_t> d_values_in, BufferView<uint32_t> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;

    static void SortPairsDescending(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<int64_t> d_values_in, BufferView<int64_t> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;
    static UCommand SortPairsDescending(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<int64_t> d_values_in, BufferView<int64_t> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;

    static void SortPairsDescending(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<uint64_t> d_values_in, BufferView<uint64_t> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;
    static UCommand SortPairsDescending(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<uint64_t> d_values_in, BufferView<uint64_t> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;

    static void SortPairsDescending(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<float> d_values_in, BufferView<float> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;
    static UCommand SortPairsDescending(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<float> d_values_in, BufferView<float> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;

    static void SortPairsDescending(size_t &temp_storage_size, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<double> d_values_in, BufferView<double> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;
    static UCommand SortPairsDescending(BufferView<int> d_temp_storage, BufferView<int32_t> d_keys_in, BufferView<int32_t> d_keys_out, BufferView<double> d_values_in, BufferView<double> d_values_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;

    static void SortKeys(size_t &temp_storage_size, const int32_t *d_keys_in, int32_t *d_keys_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;
    static UCommand SortKeys(BufferView<int> d_temp_storage, const int32_t *d_keys_in, int32_t *d_keys_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;

    static void SortKeysDescending(size_t &temp_storage_size, const int32_t *d_keys_in, int32_t *d_keys_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;
    static UCommand SortKeysDescending(BufferView<int> d_temp_storage, const int32_t *d_keys_in, int32_t *d_keys_out, int num_items, int begin_bit = 0, int end_bit = sizeof(int32_t) * 8) noexcept;
};
}// namespace luisa::compute::cuda::lcub
//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "device_merge_sort.h"
#include "dcub_utils.cuh"
#include <cub/device/device_merge_sort.cuh>

namespace luisa::compute::cuda::dcub {
// DOC:  https://nvlabs.github.io/cub/structcub_1_1_device_merge_sort.html
cudaError_t DeviceMergeSort::SortPairs(void *d_temp_storage, size_t &temp_storage_bytes, int32_t *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_items, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::SortPairs(void *d_temp_storage, size_t &temp_storage_bytes, uint32_t *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_items, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::SortPairs(void *d_temp_storage, size_t &temp_storage_bytes, int64_t *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_items, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::SortPairs(void *d_temp_storage, size_t &temp_storage_bytes, uint64_t *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_items, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::SortPairs(void *d_temp_storage, size_t &temp_storage_bytes, float *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_items, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::SortPairs(void *d_temp_storage, size_t &temp_storage_bytes, double *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_items, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::SortPairsCopy(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_input_keys, const int32_t *d_input_items, int32_t *d_output_keys, int32_t *d_output_items, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortPairsCopy(d_temp_storage, temp_storage_bytes, d_input_keys, d_input_items, d_output_keys, d_output_items, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::SortPairsCopy(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_input_keys, const int32_t *d_input_items, uint32_t *d_output_keys, int32_t *d_output_items, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortPairsCopy(d_temp_storage, temp_storage_bytes, d_input_keys, d_input_items, d_output_keys, d_output_items, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::SortPairsCopy(void *d_temp_storage, size_t &temp_storage_bytes, const int64_t *d_input_keys, const int32_t *d_input_items, int64_t *d_output_keys, int32_t *d_output_items, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortPairsCopy(d_temp_storage, temp_storage_bytes, d_input_keys, d_input_items, d_output_keys, d_output_items, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::SortPairsCopy(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_input_keys, const int32_t *d_input_items, uint64_t *d_output_keys, int32_t *d_output_items, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortPairsCopy(d_temp_storage, temp_storage_bytes, d_input_keys, d_input_items, d_output_keys, d_output_items, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::SortPairsCopy(void *d_temp_storage, size_t &temp_storage_bytes, const float *d_input_keys, const int32_t *d_input_items, float *d_output_keys, int32_t *d_output_items, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortPairsCopy(d_temp_storage, temp_storage_bytes, d_input_keys, d_input_items, d_output_keys, d_output_items, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::SortPairsCopy(void *d_temp_storage, size_t &temp_storage_bytes, const double *d_input_keys, const int32_t *d_input_items, double *d_output_keys, int32_t *d_output_items, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortPairsCopy(d_temp_storage, temp_storage_bytes, d_input_keys, d_input_items, d_output_keys, d_output_items, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::SortKeys(void *d_temp_storage, size_t &temp_storage_bytes, int32_t *d_keys, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::SortKeys(void *d_temp_storage, size_t &temp_storage_bytes, uint32_t *d_keys, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::SortKeys(void *d_temp_storage, size_t &temp_storage_bytes, int64_t *d_keys, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::SortKeys(void *d_temp_storage, size_t &temp_storage_bytes, uint64_t *d_keys, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::SortKeys(void *d_temp_storage, size_t &temp_storage_bytes, float *d_keys, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::SortKeys(void *d_temp_storage, size_t &temp_storage_bytes, double *d_keys, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::SortKeysCopy(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_input_keys, int32_t *d_output_keys, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortKeysCopy(d_temp_storage, temp_storage_bytes, d_input_keys, d_output_keys, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::SortKeysCopy(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_input_keys, uint32_t *d_output_keys, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortKeysCopy(d_temp_storage, temp_storage_bytes, d_input_keys, d_output_keys, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::SortKeysCopy(void *d_temp_storage, size_t &temp_storage_bytes, const int64_t *d_input_keys, int64_t *d_output_keys, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortKeysCopy(d_temp_storage, temp_storage_bytes, d_input_keys, d_output_keys, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::SortKeysCopy(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_input_keys, uint64_t *d_output_keys, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortKeysCopy(d_temp_storage, temp_storage_bytes, d_input_keys, d_output_keys, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::SortKeysCopy(void *d_temp_storage, size_t &temp_storage_bytes, const float *d_input_keys, float *d_output_keys, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortKeysCopy(d_temp_storage, temp_storage_bytes, d_input_keys, d_output_keys, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::SortKeysCopy(void *d_temp_storage, size_t &temp_storage_bytes, const double *d_input_keys, double *d_output_keys, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::SortKeysCopy(d_temp_storage, temp_storage_bytes, d_input_keys, d_output_keys, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::StableSortPairs(void *d_temp_storage, size_t &temp_storage_bytes, int32_t *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::StableSortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_items, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::StableSortPairs(void *d_temp_storage, size_t &temp_storage_bytes, uint32_t *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::StableSortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_items, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::StableSortPairs(void *d_temp_storage, size_t &temp_storage_bytes, int64_t *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::StableSortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_items, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::StableSortPairs(void *d_temp_storage, size_t &temp_storage_bytes, uint64_t *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::StableSortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_items, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::StableSortPairs(void *d_temp_storage, size_t &temp_storage_bytes, float *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::StableSortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_items, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::StableSortPairs(void *d_temp_storage, size_t &temp_storage_bytes, double *d_keys, int32_t *d_items, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::StableSortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_items, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::StableSortKeys(void *d_temp_storage, size_t &temp_storage_bytes, int32_t *d_keys, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::StableSortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::StableSortKeys(void *d_temp_storage, size_t &temp_storage_bytes, uint32_t *d_keys, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::StableSortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::StableSortKeys(void *d_temp_storage, size_t &temp_storage_bytes, int64_t *d_keys, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::StableSortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::StableSortKeys(void *d_temp_storage, size_t &temp_storage_bytes, uint64_t *d_keys, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::StableSortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::StableSortKeys(void *d_temp_storage, size_t &temp_storage_bytes, float *d_keys, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::StableSortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items, op, stream);
    });
}

cudaError_t DeviceMergeSort::StableSortKeys(void *d_temp_storage, size_t &temp_storage_bytes, double *d_keys, int num_items, BinaryOperator compare_op, cudaStream_t stream) {
    return op_mapper(compare_op, [&](auto op) {
        return ::cub::DeviceMergeSort::StableSortKeys(d_temp_storage, temp_storage_bytes, d_keys, num_items, op, stream);
    });
}
}// namespace luisa::compute::cuda::dcub

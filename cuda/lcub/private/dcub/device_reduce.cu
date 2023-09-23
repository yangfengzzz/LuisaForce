//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "device_reduce.h"
#include "dcub_utils.cuh"
#include <cub/device/device_reduce.cuh>

namespace luisa::compute::cuda::dcub {
// DOC:  https://nvlabs.github.io/cub/structcub_1_1_device_reduce.html
cudaError_t DeviceReduce::Sum(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_in, int32_t *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}

cudaError_t DeviceReduce::Sum(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_in, uint32_t *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}

cudaError_t DeviceReduce::Sum(void *d_temp_storage, size_t &temp_storage_bytes, const int64_t *d_in, int64_t *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}

cudaError_t DeviceReduce::Sum(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_in, uint64_t *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}

cudaError_t DeviceReduce::Sum(void *d_temp_storage, size_t &temp_storage_bytes, const float *d_in, float *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}

cudaError_t DeviceReduce::Sum(void *d_temp_storage, size_t &temp_storage_bytes, const double *d_in, double *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}

cudaError_t DeviceReduce::Max(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_in, int32_t *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}

cudaError_t DeviceReduce::Max(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_in, uint32_t *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}

cudaError_t DeviceReduce::Max(void *d_temp_storage, size_t &temp_storage_bytes, const int64_t *d_in, int64_t *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}

cudaError_t DeviceReduce::Max(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_in, uint64_t *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}

cudaError_t DeviceReduce::Max(void *d_temp_storage, size_t &temp_storage_bytes, const float *d_in, float *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}

cudaError_t DeviceReduce::Max(void *d_temp_storage, size_t &temp_storage_bytes, const double *d_in, double *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}

cudaError_t DeviceReduce::Min(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_in, int32_t *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}

cudaError_t DeviceReduce::Min(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_in, uint32_t *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}

cudaError_t DeviceReduce::Min(void *d_temp_storage, size_t &temp_storage_bytes, const int64_t *d_in, int64_t *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}

cudaError_t DeviceReduce::Min(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_in, uint64_t *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}

cudaError_t DeviceReduce::Min(void *d_temp_storage, size_t &temp_storage_bytes, const float *d_in, float *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}

cudaError_t DeviceReduce::Min(void *d_temp_storage, size_t &temp_storage_bytes, const double *d_in, double *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}

cudaError_t DeviceReduce::ArgMin(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_in, KeyValuePair<int32_t, int32_t> *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, (::cub::KeyValuePair<int32_t, int32_t> *)d_out, num_items, stream);
}

cudaError_t DeviceReduce::ArgMin(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_in, KeyValuePair<int32_t, uint32_t> *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, (::cub::KeyValuePair<int32_t, uint32_t> *)d_out, num_items, stream);
}

cudaError_t DeviceReduce::ArgMin(void *d_temp_storage, size_t &temp_storage_bytes, const int64_t *d_in, KeyValuePair<int32_t, int64_t> *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, (::cub::KeyValuePair<int32_t, int64_t> *)d_out, num_items, stream);
}

cudaError_t DeviceReduce::ArgMin(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_in, KeyValuePair<int32_t, uint64_t> *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, (::cub::KeyValuePair<int32_t, uint64_t> *)d_out, num_items, stream);
}

cudaError_t DeviceReduce::ArgMin(void *d_temp_storage, size_t &temp_storage_bytes, const float *d_in, KeyValuePair<int32_t, float> *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, (::cub::KeyValuePair<int32_t, float> *)d_out, num_items, stream);
}

cudaError_t DeviceReduce::ArgMin(void *d_temp_storage, size_t &temp_storage_bytes, const double *d_in, KeyValuePair<int32_t, double> *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, (::cub::KeyValuePair<int32_t, double> *)d_out, num_items, stream);
}

cudaError_t DeviceReduce::ArgMax(void *d_temp_storage, size_t &temp_storage_bytes, const int32_t *d_in, KeyValuePair<int32_t, int32_t> *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, (::cub::KeyValuePair<int32_t, int32_t> *)d_out, num_items, stream);
}

cudaError_t DeviceReduce::ArgMax(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_in, KeyValuePair<int32_t, uint32_t> *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, (::cub::KeyValuePair<int32_t, uint32_t> *)d_out, num_items, stream);
}

cudaError_t DeviceReduce::ArgMax(void *d_temp_storage, size_t &temp_storage_bytes, const int64_t *d_in, KeyValuePair<int32_t, int64_t> *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, (::cub::KeyValuePair<int32_t, int64_t> *)d_out, num_items, stream);
}

cudaError_t DeviceReduce::ArgMax(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_in, KeyValuePair<int32_t, uint64_t> *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, (::cub::KeyValuePair<int32_t, uint64_t> *)d_out, num_items, stream);
}

cudaError_t DeviceReduce::ArgMax(void *d_temp_storage, size_t &temp_storage_bytes, const float *d_in, KeyValuePair<int32_t, float> *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, (::cub::KeyValuePair<int32_t, float> *)d_out, num_items, stream);
}

cudaError_t DeviceReduce::ArgMax(void *d_temp_storage, size_t &temp_storage_bytes, const double *d_in, KeyValuePair<int32_t, double> *d_out, int num_items, cudaStream_t stream) {
    return ::cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, d_in, (::cub::KeyValuePair<int32_t, double> *)d_out, num_items, stream);
}
}// namespace luisa::compute::cuda::dcub

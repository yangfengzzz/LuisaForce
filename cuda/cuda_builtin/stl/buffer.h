//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "math/cuda_vec.h"

namespace wp {
template<typename T>
struct Buffer {
    T *__restrict__ ptr;
    size_t size_bytes;
};

template<typename T>
struct Buffer<const T> {
    const T *__restrict__ ptr;
    size_t size_bytes;
    Buffer(Buffer<T> buffer) noexcept
        : ptr{buffer.ptr}, size_bytes{buffer.size_bytes} {}
    Buffer() noexcept = default;
};

template<typename T>
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto buffer_size(Buffer<T> buffer) noexcept {
    return buffer.size_bytes / sizeof(T);
}

template<typename T, typename Index>
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto buffer_read(Buffer<T> buffer, Index index) noexcept {
#ifdef LUISA_DEBUG
    check_in_bounds(index, buffer_size(buffer));
#endif
    return buffer.ptr[index];
}

template<typename T, typename Index>
CUDA_CALLABLE_DEVICE inline void buffer_write(Buffer<T> buffer, Index index, T value) noexcept {
#ifdef LUISA_DEBUG
    check_in_bounds(index, buffer_size(buffer));
#endif
    buffer.ptr[index] = value;
}

template<typename T>
struct alignas(alignof(T) < 4u ? 4u : alignof(T)) Pack {
    T value;
};

template<typename T>
CUDA_CALLABLE_DEVICE inline void pack_to(const T &x, Buffer<wp_uint> array, wp_uint idx) noexcept {
    constexpr wp_uint N = (sizeof(T) + 3u) / 4u;
    if constexpr (alignof(T) < 4u) {
        // too small to be aligned to 4 bytes
        Pack<T> pack{};
        pack.value = x;
        auto data = reinterpret_cast<const wp_uint *>(&pack);
#pragma unroll
        for (auto i = 0u; i < N; i++) {
            array.ptr[idx + i] = data[i];
        }
    } else {
        // safe to reinterpret the pointer as wp_uint *
        auto data = reinterpret_cast<const wp_uint *>(&x);
#pragma unroll
        for (auto i = 0u; i < N; i++) {
            array.ptr[idx + i] = data[i];
        }
    }
}

template<typename T>
[[nodiscard]] CUDA_CALLABLE_DEVICE inline T unpack_from(Buffer<wp_uint> array, wp_uint idx) noexcept {
    if constexpr (alignof(T) <= 4u) {
        // safe to reinterpret the pointer as T *
        auto data = reinterpret_cast<const T *>(&array.ptr[idx]);
        return *data;
    } else {
        // copy to a temporary aligned buffer to avoid unaligned access
        constexpr wp_uint N = (sizeof(T) + 3u) / 4u;
        Pack<T> x{};
        auto data = reinterpret_cast<wp_uint *>(&x);
#pragma unroll
        for (auto i = 0u; i < N; i++) {
            data[i] = array.ptr[idx + i];
        }
        return x.value;
    }
}

template<typename T>
[[nodiscard]] CUDA_CALLABLE_DEVICE inline T byte_buffer_read(Buffer<const wp_byte> buffer, wp_ulong offset) noexcept {
    auto address = reinterpret_cast<wp_ulong>(buffer.ptr + offset);
#ifdef LUISA_DEBUG
    check_in_bounds(offset + sizeof(T), buffer_size(buffer));
    assert(address % alignof(T) == 0u && "unaligned access");
#endif
    return *reinterpret_cast<T *>(address);
}

template<typename T>
CUDA_CALLABLE_DEVICE inline void byte_buffer_write(Buffer<wp_byte> buffer, wp_ulong offset, T value) noexcept {
    auto address = reinterpret_cast<wp_ulong>(buffer.ptr + offset);
#ifdef LUISA_DEBUG
    check_in_bounds(offset + sizeof(T), buffer_size(buffer));
    assert(address % alignof(T) == 0u && "unaligned access");
#endif
    *reinterpret_cast<T *>(address) = value;
}

[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto byte_buffer_size(Buffer<const wp_byte> buffer) noexcept {
    return buffer_size(buffer);
}

}// namespace wp
//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "math/cuda_mat.h"

namespace wp {
[[nodiscard]] CUDA_CALLABLE_DEVICE inline bool isinf_impl(wp_float x) noexcept {
    auto u = __float_as_int(x);
    return u == 0x7f800000u | u == 0xff800000u;
}
[[nodiscard]] CUDA_CALLABLE_DEVICE inline bool isnan_impl(wp_float x) noexcept {
    auto u = __float_as_int(x);
    return ((u & 0x7F800000u) == 0x7F800000u) & ((u & 0x7FFFFFu) != 0u);
}

[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto isinf(wp_float x) noexcept { return isinf_impl(x); }
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto isinf(wp_float2 x) noexcept { return wp_bool2(isinf_impl(x.x()), isinf_impl(x.y())); }
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto isinf(wp_float3 x) noexcept { return wp_bool3(isinf_impl(x.x()), isinf_impl(x.y()), isinf_impl(x.z())); }
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto isinf(wp_float4 x) noexcept { return wp_bool4(isinf_impl(x.x()), isinf_impl(x.y()), isinf_impl(x.z()), isinf_impl(x.w())); }

[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto isnan(wp_float x) noexcept { return isnan_impl(x); }
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto isnan(wp_float2 x) noexcept { return wp_bool2(isnan_impl(x.x()), isnan_impl(x.y())); }
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto isnan(wp_float3 x) noexcept { return wp_bool3(isnan_impl(x.x()), isnan_impl(x.y()), isnan_impl(x.z())); }
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto isnan(wp_float4 x) noexcept { return wp_bool4(isnan_impl(x.x()), isnan_impl(x.y()), isnan_impl(x.z()), isnan_impl(x.w())); }

[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto clz(wp_uint x) noexcept { return __clz(x); }
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto clz(wp_uint2 x) noexcept { return wp_uint2(__clz(x.x()), __clz(x.y())); }
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto clz(wp_uint3 x) noexcept { return wp_uint3(__clz(x.x()), __clz(x.y()), __clz(x.z())); }
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto clz(wp_uint4 x) noexcept { return wp_uint4(__clz(x.x()), __clz(x.y()), __clz(x.z()), __clz(x.w())); }

[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto popcount(wp_uint x) noexcept { return __popc(x); }
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto popcount(wp_uint2 x) noexcept { return wp_uint2(__popc(x.x()), __popc(x.y())); }
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto popcount(wp_uint3 x) noexcept { return wp_uint3(__popc(x.x()), __popc(x.y()), __popc(x.z())); }
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto popcount(wp_uint4 x) noexcept { return wp_uint4(__popc(x.x()), __popc(x.y()), __popc(x.z()), __popc(x.w())); }

[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto reverse(wp_uint x) noexcept { return __brev(x); }
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto reverse(wp_uint2 x) noexcept { return wp_uint2(__brev(x.x()), __brev(x.y())); }
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto reverse(wp_uint3 x) noexcept { return wp_uint3(__brev(x.x()), __brev(x.y()), __brev(x.z())); }
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto reverse(wp_uint4 x) noexcept { return wp_uint4(__brev(x.x()), __brev(x.y()), __brev(x.z()), __brev(x.w())); }

[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto ctz_impl(wp_uint x) noexcept { return (__ffs(x) - 1u) % 32u; }
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto ctz(wp_uint x) noexcept { return ctz_impl(x); }
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto ctz(wp_uint2 x) noexcept { return wp_uint2(ctz_impl(x.x()), ctz_impl(x.y())); }
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto ctz(wp_uint3 x) noexcept { return wp_uint3(ctz_impl(x.x()), ctz_impl(x.y()), ctz_impl(x.z())); }
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto ctz(wp_uint4 x) noexcept { return wp_uint4(ctz_impl(x.x()), ctz_impl(x.y()), ctz_impl(x.z()), ctz_impl(x.w())); }

// warp intrinsics
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_lane_id() noexcept {
    wp_uint ret;
    asm("mov.u32 %0, %laneid;"
        : "=r"(ret));
    return ret;
}

[[nodiscard]] CUDA_CALLABLE_DEVICE constexpr auto warp_size() noexcept {
    return static_cast<wp_uint>(warpSize);
}

#define WARP_FULL_MASK 0xffff'ffffu
#define WARP_ACTIVE_MASK __activemask()

[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_first_active_lane() noexcept {
    return __ffs(WARP_ACTIVE_MASK) - 1u;
}

[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_is_first_active_lane() noexcept {
    return warp_first_active_lane() == warp_lane_id();
}

#define WARP_ALL_EQ_SCALAR(T)                                                            \
    [[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_all_equal(T x) noexcept { \
        auto mask = WARP_ACTIVE_MASK;                                                    \
        auto pred = 0;                                                                   \
        __match_all_sync(mask, x, &pred);                                                \
        return pred != 0;                                                                \
    }

#define WARP_ALL_EQ_VECTOR2(T)                                                              \
    [[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_all_equal(T##2 v) noexcept { \
        return wp_bool2(warp_active_all_equal(v.x()),                                       \
                        warp_active_all_equal(v.y()));                                      \
    }

#define WARP_ALL_EQ_VECTOR3(T)                                                              \
    [[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_all_equal(T##3 v) noexcept { \
        return wp_bool3(warp_active_all_equal(v.x()),                                       \
                        warp_active_all_equal(v.y()),                                       \
                        warp_active_all_equal(v.z()));                                      \
    }

#define WARP_ALL_EQ_VECTOR4(T)                                                              \
    [[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_all_equal(T##4 v) noexcept { \
        return wp_bool4(warp_active_all_equal(v.x()),                                       \
                        warp_active_all_equal(v.y()),                                       \
                        warp_active_all_equal(v.z()),                                       \
                        warp_active_all_equal(v.w()));                                      \
    }

#define WARP_ALL_EQ(T)     \
    WARP_ALL_EQ_SCALAR(T)  \
    WARP_ALL_EQ_VECTOR2(T) \
    WARP_ALL_EQ_VECTOR3(T) \
    WARP_ALL_EQ_VECTOR4(T)

WARP_ALL_EQ(wp_bool)
WARP_ALL_EQ(wp_short)
WARP_ALL_EQ(wp_ushort)
WARP_ALL_EQ(wp_int)
WARP_ALL_EQ(wp_uint)
WARP_ALL_EQ(wp_float)
WARP_ALL_EQ(wp_long)
WARP_ALL_EQ(wp_ulong)
//WARP_ALL_EQ(half)// TODO
//WARP_ALL_EQ(double)// TODO

#undef WARP_ALL_EQ_SCALAR
#undef WARP_ALL_EQ_VECTOR2
#undef WARP_ALL_EQ_VECTOR3
#undef WARP_ALL_EQ_VECTOR4
#undef WARP_ALL_EQ

template<typename T, typename F>
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_reduce_impl(T x, F f) noexcept {
    auto mask = WARP_ACTIVE_MASK;
    auto lane = warp_lane_id();
    if (auto y = __shfl_xor_sync(mask, x, 0x10u); mask & (1u << (lane ^ 0x10u))) { x = f(x, y); }
    if (auto y = __shfl_xor_sync(mask, x, 0x08u); mask & (1u << (lane ^ 0x08u))) { x = f(x, y); }
    if (auto y = __shfl_xor_sync(mask, x, 0x04u); mask & (1u << (lane ^ 0x04u))) { x = f(x, y); }
    if (auto y = __shfl_xor_sync(mask, x, 0x02u); mask & (1u << (lane ^ 0x02u))) { x = f(x, y); }
    if (auto y = __shfl_xor_sync(mask, x, 0x01u); mask & (1u << (lane ^ 0x01u))) { x = f(x, y); }
    return x;
}

template<typename T>
[[nodiscard]] CUDA_CALLABLE_DEVICE constexpr T bit_and(T x, T y) noexcept { return x & y; }

template<typename T>
[[nodiscard]] CUDA_CALLABLE_DEVICE constexpr T bit_or(T x, T y) noexcept { return x | y; }

template<typename T>
[[nodiscard]] CUDA_CALLABLE_DEVICE constexpr T bit_xor(T x, T y) noexcept { return x ^ y; }

#define WARP_REDUCE_BIT_SCALAR_FALLBACK(op, T)                                            \
    [[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_bit_##op(##T x) noexcept { \
        return static_cast<##T>(warp_active_reduce_impl(                                  \
            x, [](##T a, ##T b) noexcept { return bit_##op(a, b); }));                    \
    }

#if __CUDA_ARCH__ >= 800
#define WARP_REDUCE_BIT_SCALAR(op, T)                                                     \
    [[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_bit_##op(##T x) noexcept { \
        return static_cast<##T>(__reduce_##op##_sync(WARP_ACTIVE_MASK,                    \
                                                     static_cast<wp_uint>(x)));           \
    }
#else
#define WARP_REDUCE_BIT_SCALAR(op, T) WARP_REDUCE_BIT_SCALAR_FALLBACK(op, T)
#endif

WARP_REDUCE_BIT_SCALAR(and, wp_uint)
WARP_REDUCE_BIT_SCALAR(or, wp_uint)
WARP_REDUCE_BIT_SCALAR(xor, wp_uint)
WARP_REDUCE_BIT_SCALAR(and, wp_int)
WARP_REDUCE_BIT_SCALAR(or, wp_int)
WARP_REDUCE_BIT_SCALAR(xor, wp_int)

WARP_REDUCE_BIT_SCALAR(and, wp_ushort)
WARP_REDUCE_BIT_SCALAR(or, wp_ushort)
WARP_REDUCE_BIT_SCALAR(xor, wp_ushort)
WARP_REDUCE_BIT_SCALAR(and, wp_short)
WARP_REDUCE_BIT_SCALAR(or, wp_short)
WARP_REDUCE_BIT_SCALAR(xor, wp_short)

#undef WARP_REDUCE_BIT_SCALAR_FALLBACK
#undef WARP_REDUCE_BIT_SCALAR

#define WARP_REDUCE_BIT_VECTOR(op, T)                                                        \
    [[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_bit_##op(##T##2 v) noexcept { \
        return vec_t<2, T>(warp_active_bit_##op(v.x()),                                      \
                           warp_active_bit_##op(v.y()));                                     \
    }                                                                                        \
    [[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_bit_##op(##T##3 v) noexcept { \
        return vec_t<3, T>(warp_active_bit_##op(v.x()),                                      \
                           warp_active_bit_##op(v.y()),                                      \
                           warp_active_bit_##op(v.z()));                                     \
    }                                                                                        \
    [[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_bit_##op(##T##4 v) noexcept { \
        return vec_t<4, T>(warp_active_bit_##op(v.x()),                                      \
                           warp_active_bit_##op(v.y()),                                      \
                           warp_active_bit_##op(v.z()),                                      \
                           warp_active_bit_##op(v.w()));                                     \
    }

WARP_REDUCE_BIT_VECTOR(and, wp_uint)
WARP_REDUCE_BIT_VECTOR(or, wp_uint)
WARP_REDUCE_BIT_VECTOR(xor, wp_uint)
WARP_REDUCE_BIT_VECTOR(and, wp_int)
WARP_REDUCE_BIT_VECTOR(or, wp_int)
WARP_REDUCE_BIT_VECTOR(xor, wp_int)

#undef WARP_REDUCE_BIT_VECTOR

[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_bit_mask(bool pred) noexcept {
    return wp_uint4(__ballot_sync(WARP_ACTIVE_MASK, pred), 0u, 0u, 0u);
}

[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_count_bits(bool pred) noexcept {
    return popcount(__ballot_sync(WARP_ACTIVE_MASK, pred));
}

[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_all(bool pred) noexcept {
    return static_cast<bool>(__all_sync(WARP_ACTIVE_MASK, pred));
}

[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_any(bool pred) noexcept {
    return static_cast<bool>(__any_sync(WARP_ACTIVE_MASK, pred));
}

[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_prefix_mask() noexcept {
    wp_uint ret;
    asm("mov.u32 %0, %lanemask_lt;"
        : "=r"(ret));
    return ret;
}

[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_prefix_count_bits(bool pred) noexcept {
    return popcount(__ballot_sync(WARP_ACTIVE_MASK, pred) & warp_prefix_mask());
}

#define WARP_READ_LANE_SCALAR(T)                                                               \
    [[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_read_lane(##T x, wp_uint i) noexcept { \
        return static_cast<##T>(__shfl_sync(WARP_ACTIVE_MASK, x, i));                          \
    }

#define WARP_READ_LANE_VECTOR2(T)                                                                 \
    [[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_read_lane(##T##2 v, wp_uint i) noexcept { \
        return vec_t<2, T>(warp_read_lane(v.x(), i),                                              \
                           warp_read_lane(v.y(), i));                                             \
    }

#define WARP_READ_LANE_VECTOR3(T)                                                                 \
    [[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_read_lane(##T##3 v, wp_uint i) noexcept { \
        return vec_t<3, T>(warp_read_lane(v.x(), i),                                              \
                           warp_read_lane(v.y(), i),                                              \
                           warp_read_lane(v.z(), i));                                             \
    }

#define WARP_READ_LANE_VECTOR4(T)                                                                 \
    [[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_read_lane(##T##4 v, wp_uint i) noexcept { \
        return vec_t<4, T>(warp_read_lane(v.x(), i),                                              \
                           warp_read_lane(v.y(), i),                                              \
                           warp_read_lane(v.z(), i),                                              \
                           warp_read_lane(v.w(), i));                                             \
    }

#define WARP_READ_LANE(T)     \
    WARP_READ_LANE_SCALAR(T)  \
    WARP_READ_LANE_VECTOR2(T) \
    WARP_READ_LANE_VECTOR3(T) \
    WARP_READ_LANE_VECTOR4(T)

WARP_READ_LANE(wp_bool)
WARP_READ_LANE(wp_short)
WARP_READ_LANE(wp_ushort)
WARP_READ_LANE(wp_int)
WARP_READ_LANE(wp_uint)
WARP_READ_LANE(wp_float)
//WARP_READ_LANE(half)// TODO
//WARP_READ_LANE(double)// TODO

#undef WARP_READ_LANE_SCALAR
#undef WARP_READ_LANE_VECTOR2
#undef WARP_READ_LANE_VECTOR3
#undef WARP_READ_LANE_VECTOR4
#undef WARP_READ_LANE

[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_read_lane(float2x2 m, wp_uint i) noexcept {
    return float2x2(warp_read_lane(m[0], i),
                    warp_read_lane(m[1], i));
}

[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_read_lane(float3x3 m, wp_uint i) noexcept {
    return float3x3(warp_read_lane(m[0], i),
                    warp_read_lane(m[1], i),
                    warp_read_lane(m[2], i));
}

[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_read_lane(float4x4 m, wp_uint i) noexcept {
    return float4x4(warp_read_lane(m[0], i),
                    warp_read_lane(m[1], i),
                    warp_read_lane(m[2], i),
                    warp_read_lane(m[3], i));
}

template<typename T>
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_read_first_active_lane(T x) noexcept {
    return warp_read_lane(x, warp_first_active_lane());
}

template<typename T>
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_min_impl(T x) noexcept {
    return warp_active_reduce_impl(x, [](T a, T b) noexcept { return min(a, b); });
}
template<typename T>
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_max_impl(T x) noexcept {
    return warp_active_reduce_impl(x, [](T a, T b) noexcept { return max(a, b); });
}
template<typename T>
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_sum_impl(T x) noexcept {
    return warp_active_reduce_impl(x, [](T a, T b) noexcept { return a + b; });
}
template<typename T>
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_product_impl(T x) noexcept {
    return warp_active_reduce_impl(x, [](T a, T b) noexcept { return a * b; });
}

#define WARP_ACTIVE_REDUCE_SCALAR(op, T)                                              \
    [[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_##op(##T x) noexcept { \
        return warp_active_##op##_impl<##T>(x);                                       \
    }

#if __CUDA_ARCH__ >= 800
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_min(wp_uint x) noexcept {
    return __reduce_min_sync(WARP_ACTIVE_MASK, x);
}
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_max(wp_uint x) noexcept {
    return __reduce_max_sync(WARP_ACTIVE_MASK, x);
}
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_sum(wp_uint x) noexcept {
    return __reduce_add_sync(WARP_ACTIVE_MASK, x);
}
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_min(wp_int x) noexcept {
    return __reduce_min_sync(WARP_ACTIVE_MASK, x);
}
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_max(wp_int x) noexcept {
    return __reduce_max_sync(WARP_ACTIVE_MASK, x);
}
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_sum(wp_int x) noexcept {
    return __reduce_add_sync(WARP_ACTIVE_MASK, x);
}
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_min(wp_ushort x) noexcept {
    return static_cast<wp_ushort>(__reduce_min_sync(WARP_ACTIVE_MASK, static_cast<wp_uint>(x)));
}
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_max(wp_ushort x) noexcept {
    return static_cast<wp_ushort>(__reduce_max_sync(WARP_ACTIVE_MASK, static_cast<wp_uint>(x)));
}
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_sum(wp_ushort x) noexcept {
    return static_cast<wp_ushort>(__reduce_add_sync(WARP_ACTIVE_MASK, static_cast<wp_uint>(x)));
}
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_min(wp_short x) noexcept {
    return static_cast<short>(__reduce_min_sync(WARP_ACTIVE_MASK, static_cast<wp_int>(x)));
}
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_max(wp_short x) noexcept {
    return static_cast<short>(__reduce_max_sync(WARP_ACTIVE_MASK, static_cast<wp_int>(x)));
}
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_sum(wp_short x) noexcept {
    return static_cast<short>(__reduce_add_sync(WARP_ACTIVE_MASK, static_cast<wp_int>(x)));
}
#else
WARP_ACTIVE_REDUCE_SCALAR(min, wp_uint)
WARP_ACTIVE_REDUCE_SCALAR(max, wp_uint)
WARP_ACTIVE_REDUCE_SCALAR(sum, wp_uint)
WARP_ACTIVE_REDUCE_SCALAR(min, wp_int)
WARP_ACTIVE_REDUCE_SCALAR(max, wp_int)
WARP_ACTIVE_REDUCE_SCALAR(sum, wp_int)
WARP_ACTIVE_REDUCE_SCALAR(min, wp_ushort)
WARP_ACTIVE_REDUCE_SCALAR(max, wp_ushort)
WARP_ACTIVE_REDUCE_SCALAR(sum, wp_ushort)
WARP_ACTIVE_REDUCE_SCALAR(min, wp_short)
WARP_ACTIVE_REDUCE_SCALAR(max, wp_short)
WARP_ACTIVE_REDUCE_SCALAR(sum, wp_short)
#endif

WARP_ACTIVE_REDUCE_SCALAR(product, wp_uint)
WARP_ACTIVE_REDUCE_SCALAR(product, wp_int)
WARP_ACTIVE_REDUCE_SCALAR(product, wp_ushort)
WARP_ACTIVE_REDUCE_SCALAR(product, wp_short)
WARP_ACTIVE_REDUCE_SCALAR(min, wp_float)
WARP_ACTIVE_REDUCE_SCALAR(max, wp_float)
WARP_ACTIVE_REDUCE_SCALAR(sum, wp_float)
WARP_ACTIVE_REDUCE_SCALAR(product, wp_float)
// TODO: half and double
// WARP_ACTIVE_REDUCE_SCALAR(min, half)
// WARP_ACTIVE_REDUCE_SCALAR(max, half)
// WARP_ACTIVE_REDUCE_SCALAR(sum, half)
// WARP_ACTIVE_REDUCE_SCALAR(product, half)
// WARP_ACTIVE_REDUCE_SCALAR(min, double)
// WARP_ACTIVE_REDUCE_SCALAR(max, double)
// WARP_ACTIVE_REDUCE_SCALAR(sum, double)
// WARP_ACTIVE_REDUCE_SCALAR(product, double)

#undef WARP_ACTIVE_REDUCE_SCALAR

#define WARP_ACTIVE_REDUCE_VECTOR2(op, T)                                                \
    [[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_##op(##T##2 v) noexcept { \
        return vec_t<2, T>(warp_active_##op(v.x()),                                      \
                           warp_active_##op(v.y()));                                     \
    }

#define WARP_ACTIVE_REDUCE_VECTOR3(op, T)                                                \
    [[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_##op(##T##3 v) noexcept { \
        return vec_t<3, T>(warp_active_##op(v.x()),                                      \
                           warp_active_##op(v.y()),                                      \
                           warp_active_##op(v.z()));                                     \
    }

#define WARP_ACTIVE_REDUCE_VECTOR4(op, T)                                                \
    [[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_active_##op(##T##4 v) noexcept { \
        return vec_t<4, T>(warp_active_##op(v.x()),                                      \
                           warp_active_##op(v.y()),                                      \
                           warp_active_##op(v.z()),                                      \
                           warp_active_##op(v.w()));                                     \
    }

#define WARP_ACTIVE_REDUCE(T)              \
    WARP_ACTIVE_REDUCE_VECTOR2(min, T)     \
    WARP_ACTIVE_REDUCE_VECTOR3(min, T)     \
    WARP_ACTIVE_REDUCE_VECTOR4(min, T)     \
    WARP_ACTIVE_REDUCE_VECTOR2(max, T)     \
    WARP_ACTIVE_REDUCE_VECTOR3(max, T)     \
    WARP_ACTIVE_REDUCE_VECTOR4(max, T)     \
    WARP_ACTIVE_REDUCE_VECTOR2(sum, T)     \
    WARP_ACTIVE_REDUCE_VECTOR3(sum, T)     \
    WARP_ACTIVE_REDUCE_VECTOR4(sum, T)     \
    WARP_ACTIVE_REDUCE_VECTOR2(product, T) \
    WARP_ACTIVE_REDUCE_VECTOR3(product, T) \
    WARP_ACTIVE_REDUCE_VECTOR4(product, T)

WARP_ACTIVE_REDUCE(wp_uint)
WARP_ACTIVE_REDUCE(wp_int)
WARP_ACTIVE_REDUCE(wp_ushort)
WARP_ACTIVE_REDUCE(wp_short)
WARP_ACTIVE_REDUCE(wp_float)
//WARP_ACTIVE_REDUCE(half)// TODO
//WARP_ACTIVE_REDUCE(double)// TODO

#undef WARP_ACTIVE_REDUCE_VECTOR2
#undef WARP_ACTIVE_REDUCE_VECTOR3
#undef WARP_ACTIVE_REDUCE_VECTOR4
#undef WARP_ACTIVE_REDUCE

[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_prev_active_lane() noexcept {
    auto mask = 0u;
    asm("mov.u32 %0, %lanemask_lt;"
        : "=r"(mask));
    return (warp_size() - 1u) - __clz(WARP_ACTIVE_MASK & mask);
}

template<typename T, typename F>
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_prefix_reduce_impl(T x, T unit, F f) noexcept {
    auto mask = WARP_ACTIVE_MASK;
    auto lane = warp_lane_id();
    x = __shfl_sync(mask, x, warp_prev_active_lane());
    x = (lane == warp_first_active_lane()) ? unit : x;
    if (auto y = __shfl_up_sync(mask, x, 0x01u); lane >= 0x01u && (mask & (1u << (lane - 0x01u)))) { x = f(x, y); }
    if (auto y = __shfl_up_sync(mask, x, 0x02u); lane >= 0x02u && (mask & (1u << (lane - 0x02u)))) { x = f(x, y); }
    if (auto y = __shfl_up_sync(mask, x, 0x04u); lane >= 0x04u && (mask & (1u << (lane - 0x04u)))) { x = f(x, y); }
    if (auto y = __shfl_up_sync(mask, x, 0x08u); lane >= 0x08u && (mask & (1u << (lane - 0x08u)))) { x = f(x, y); }
    if (auto y = __shfl_up_sync(mask, x, 0x10u); lane >= 0x10u && (mask & (1u << (lane - 0x10u)))) { x = f(x, y); }
    return x;
}

template<typename T>
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_prefix_sum_impl(T x) noexcept {
    return warp_prefix_reduce_impl(x, static_cast<T>(0), [](T a, T b) noexcept { return a + b; });
}

template<typename T>
[[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_prefix_product_impl(T x) noexcept {
    return warp_prefix_reduce_impl(x, static_cast<T>(1), [](T a, T b) noexcept { return a * b; });
}

#define WARP_PREFIX_REDUCE_SCALAR(op, T)                                              \
    [[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_prefix_##op(##T x) noexcept { \
        return warp_prefix_##op##_impl<##T>(x);                                       \
    }

#define WARP_PREFIX_REDUCE_VECTOR2(op, T)                                                \
    [[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_prefix_##op(##T##2 v) noexcept { \
        return vec_t<2, T>(warp_prefix_##op(v.x()),                                      \
                           warp_prefix_##op(v.y()));                                     \
    }

#define WARP_PREFIX_REDUCE_VECTOR3(op, T)                                                \
    [[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_prefix_##op(##T##3 v) noexcept { \
        return vec_t<3, T>(warp_prefix_##op(v.x()),                                      \
                           warp_prefix_##op(v.y()),                                      \
                           warp_prefix_##op(v.z()));                                     \
    }

#define WARP_PREFIX_REDUCE_VECTOR4(op, T)                                                \
    [[nodiscard]] CUDA_CALLABLE_DEVICE inline auto warp_prefix_##op(##T##4 v) noexcept { \
        return vec_t<4, T>(warp_prefix_##op(v.x()),                                      \
                           warp_prefix_##op(v.y()),                                      \
                           warp_prefix_##op(v.z()),                                      \
                           warp_prefix_##op(v.w()));                                     \
    }

#define WARP_PREFIX_REDUCE(T)              \
    WARP_PREFIX_REDUCE_SCALAR(sum, T)      \
    WARP_PREFIX_REDUCE_SCALAR(product, T)  \
    WARP_PREFIX_REDUCE_VECTOR2(sum, T)     \
    WARP_PREFIX_REDUCE_VECTOR2(product, T) \
    WARP_PREFIX_REDUCE_VECTOR3(sum, T)     \
    WARP_PREFIX_REDUCE_VECTOR3(product, T) \
    WARP_PREFIX_REDUCE_VECTOR4(sum, T)     \
    WARP_PREFIX_REDUCE_VECTOR4(product, T)

WARP_PREFIX_REDUCE(wp_uint)
WARP_PREFIX_REDUCE(wp_int)
WARP_PREFIX_REDUCE(wp_ushort)
WARP_PREFIX_REDUCE(wp_short)
WARP_PREFIX_REDUCE(wp_float)
//WARP_PREFIX_REDUCE(half)// TODO
//WARP_PREFIX_REDUCE(double)// TODO

#undef WARP_PREFIX_REDUCE_SCALAR
#undef WARP_PREFIX_REDUCE_VECTOR2
#undef WARP_PREFIX_REDUCE_VECTOR3
#undef WARP_PREFIX_REDUCE_VECTOR4
#undef WARP_PREFIX_REDUCE
}// namespace wp
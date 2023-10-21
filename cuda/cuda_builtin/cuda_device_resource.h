#pragma once

[[nodiscard]] __device__ constexpr auto lc_infinity_float() noexcept { return __int_as_float(0x7f800000u); }
[[nodiscard]] __device__ constexpr auto lc_infinity_double() noexcept { return __longlong_as_double(0x7ff0000000000000ull); }

#if LC_NVRTC_VERSION < 110200
#define LC_CONSTANT const
#else
#define LC_CONSTANT constexpr
#endif

#if LC_NVRTC_VERSION < 110200
inline __device__ void lc_assume(bool) noexcept {}
#else
#define lc_assume(...) __builtin_assume(__VA_ARGS__)
#endif

[[noreturn]] inline void lc_trap() noexcept { asm("trap;"); }

template<typename T = void>
[[noreturn]] inline __device__ T lc_unreachable(
    const char *file, int line,
    const char *user_msg = nullptr) noexcept {
#if LC_NVRTC_VERSION < 110300 || defined(LUISA_DEBUG)
    printf("Unreachable code reached: %s. [%s:%d]\n",
           user_msg ? user_msg : "no user-specified message",
           file, line);
    lc_trap();
#else
    __builtin_unreachable();
#endif
}

#ifdef LUISA_DEBUG
#define lc_assert(x)                                    \
    do {                                                \
        if (!(x)) {                                     \
            printf("Assertion failed: %s [%s:%d:%s]\n", \
                   #x,                                  \
                   __FILE__,                            \
                   static_cast<int>(__LINE__),          \
                   __FUNCTION__);                       \
            lc_trap();                                  \
        }                                               \
    } while (false)
#define lc_check_in_bounds(size, max_size)                               \
    do {                                                                 \
        if (!((size) < (max_size))) {                                    \
            printf("Out of bounds: !(%s: %llu < %s: %llu) [%s:%d:%s]\n", \
                   #size, static_cast<size_t>(size),                     \
                   #max_size, static_cast<size_t>(max_size),             \
                   __FILE__, static_cast<int>(__LINE__),                 \
                   __FUNCTION__);                                        \
            lc_trap();                                                   \
        }                                                                \
    } while (false)
#else
inline __device__ void lc_assert(bool) noexcept {}
#endif

struct lc_half {
    unsigned short bits;
};

struct alignas(4) lc_half2 {
    lc_half x, y;
};

struct alignas(8) lc_half4 {
    lc_half x, y, z, w;
};

[[nodiscard]] __device__ inline auto lc_half_to_float(lc_half x) noexcept {
    lc_float val;
    asm("{  cvt.f32.f16 %0, %1;}\n"
        : "=f"(val)
        : "h"(x.bits));
    return val;
}

[[nodiscard]] __device__ inline auto lc_float_to_half(lc_float x) noexcept {
    lc_half val;
    asm("{  cvt.rn.f16.f32 %0, %1;}\n"
        : "=h"(val.bits)
        : "f"(x));
    return val;
}

template<size_t alignment, size_t size>
struct alignas(alignment) lc_aligned_storage {
    unsigned char data[size];
};

struct alignas(16) LCIndirectHeader {
    lc_uint size;
};

struct alignas(16) LCIndirectDispatch {
    lc_uint3 block_size;
    lc_uint4 dispatch_size_and_kernel_id;
};

struct alignas(16) LCIndirectBuffer {
    void *__restrict__ data;
    lc_uint offset;
    lc_uint capacity;

    [[nodiscard]] auto header() const noexcept {
        return reinterpret_cast<LCIndirectHeader *>(data);
    }

    [[nodiscard]] auto dispatches() const noexcept {
        return reinterpret_cast<LCIndirectDispatch *>(reinterpret_cast<lc_ulong>(data) + sizeof(LCIndirectHeader));
    }
};

void lc_indirect_set_dispatch_count(const LCIndirectBuffer buffer, lc_uint count) noexcept {
#ifdef LUISA_DEBUG
    lc_check_in_bounds(buffer.offset + count, buffer.capacity);
#endif
    buffer.header()->size = count;
}

void lc_indirect_set_dispatch_kernel(const LCIndirectBuffer buffer, lc_uint index, lc_uint3 block_size, lc_uint3 dispatch_size, lc_uint kernel_id) noexcept {
#ifdef LUISA_DEBUG
    lc_check_in_bounds(index, buffer.header()->size);
    lc_check_in_bounds(index + buffer.offset, buffer.capacity);
#endif
    buffer.dispatches()[index + buffer.offset] = LCIndirectDispatch{block_size, lc_make_uint4(dispatch_size, kernel_id)};
}

template<typename T>
struct LCBuffer {
    T *__restrict__ ptr;
    size_t size_bytes;
};

template<typename T>
struct LCBuffer<const T> {
    const T *__restrict__ ptr;
    size_t size_bytes;
    LCBuffer(LCBuffer<T> buffer) noexcept
        : ptr{buffer.ptr}, size_bytes{buffer.size_bytes} {}
    LCBuffer() noexcept = default;
};

template<typename T>
[[nodiscard]] __device__ inline auto lc_buffer_size(LCBuffer<T> buffer) noexcept {
    return buffer.size_bytes / sizeof(T);
}

template<typename T, typename Index>
[[nodiscard]] __device__ inline auto lc_buffer_read(LCBuffer<T> buffer, Index index) noexcept {
    lc_assume(__isGlobal(buffer.ptr));
#ifdef LUISA_DEBUG
    lc_check_in_bounds(index, lc_buffer_size(buffer));
#endif
    return buffer.ptr[index];
}

template<typename T, typename Index>
__device__ inline void lc_buffer_write(LCBuffer<T> buffer, Index index, T value) noexcept {
    lc_assume(__isGlobal(buffer.ptr));
#ifdef LUISA_DEBUG
    lc_check_in_bounds(index, lc_buffer_size(buffer));
#endif
    buffer.ptr[index] = value;
}

enum struct LCPixelStorage {

    BYTE1,
    BYTE2,
    BYTE4,

    SHORT1,
    SHORT2,
    SHORT4,

    INT1,
    INT2,
    INT4,

    HALF1,
    HALF2,
    HALF4,

    FLOAT1,
    FLOAT2,
    FLOAT4
};

struct alignas(16) LCSurface {
    cudaSurfaceObject_t handle;
    unsigned long long storage;
};

static_assert(sizeof(LCSurface) == 16);

template<typename A, typename B>
struct lc_is_same {
    [[nodiscard]] static constexpr auto value() noexcept { return false; };
};

template<typename A>
struct lc_is_same<A, A> {
    [[nodiscard]] static constexpr auto value() noexcept { return true; };
};

template<typename...>
struct lc_always_false {
    [[nodiscard]] static constexpr auto value() noexcept { return false; };
};

template<typename P>
[[nodiscard]] __device__ inline auto lc_texel_to_float(P x) noexcept {
    if constexpr (lc_is_same<P, char>::value()) {
        return static_cast<unsigned char>(x) * (1.0f / 255.0f);
    } else if constexpr (lc_is_same<P, short>::value()) {
        return static_cast<unsigned short>(x) * (1.0f / 65535.0f);
    } else if constexpr (lc_is_same<P, lc_half>::value()) {
        return lc_half_to_float(x);
    } else if constexpr (lc_is_same<P, lc_float>::value()) {
        return x;
    }
    return 0.0f;
}

template<typename P>
[[nodiscard]] __device__ inline auto lc_texel_to_int(P x) noexcept {
    if constexpr (lc_is_same<P, char>::value()) {
        return static_cast<lc_int>(x);
    } else if constexpr (lc_is_same<P, short>::value()) {
        return static_cast<lc_int>(x);
    } else if constexpr (lc_is_same<P, lc_int>::value()) {
        return x;
    }
    return 0;
}

template<typename P>
[[nodiscard]] __device__ inline auto lc_texel_to_uint(P x) noexcept {
    if constexpr (lc_is_same<P, char>::value()) {
        return static_cast<lc_uint>(static_cast<unsigned char>(x));
    } else if constexpr (lc_is_same<P, short>::value()) {
        return static_cast<lc_uint>(static_cast<unsigned short>(x));
    } else if constexpr (lc_is_same<P, lc_int>::value()) {
        return static_cast<lc_uint>(x);
    }
    return 0u;
}

template<typename T, typename P>
[[nodiscard]] __device__ inline auto lc_texel_read_convert(P p) noexcept {
    if constexpr (lc_is_same<T, lc_float>::value()) {
        return lc_texel_to_float<P>(p);
    } else if constexpr (lc_is_same<T, lc_int>::value()) {
        return lc_texel_to_int<P>(p);
    } else if constexpr (lc_is_same<T, lc_uint>::value()) {
        return lc_texel_to_uint<P>(p);
    } else {
        static_assert(lc_always_false<T, P>::value());
    }
}

template<typename P>
[[nodiscard]] __device__ inline auto lc_float_to_texel(lc_float x) noexcept {
    if constexpr (lc_is_same<P, char>::value()) {
        return static_cast<char>(static_cast<unsigned char>(lc_round(lc_saturate(x) * 255.0f)));
    } else if constexpr (lc_is_same<P, short>::value()) {
        return static_cast<short>(static_cast<unsigned short>(lc_round(lc_saturate(x) * 65535.0f)));
    } else if constexpr (lc_is_same<P, lc_half>::value()) {
        return lc_float_to_half(x);
    } else if constexpr (lc_is_same<P, lc_float>::value()) {
        return x;
    }
    return P{};
}

template<typename P>
[[nodiscard]] __device__ inline auto lc_int_to_texel(lc_int x) noexcept {
    if constexpr (lc_is_same<P, char>::value()) {
        return static_cast<char>(x);
    } else if constexpr (lc_is_same<P, short>::value()) {
        return static_cast<short>(x);
    } else if constexpr (lc_is_same<P, lc_int>::value()) {
        return x;
    }
    return P{};
}

template<typename P>
[[nodiscard]] __device__ inline auto lc_uint_to_texel(lc_uint x) noexcept {
    if constexpr (lc_is_same<P, char>::value()) {
        return static_cast<char>(static_cast<unsigned char>(x));
    } else if constexpr (lc_is_same<P, short>::value()) {
        return static_cast<short>(static_cast<unsigned short>(x));
    } else if constexpr (lc_is_same<P, lc_int>::value()) {
        return static_cast<lc_int>(x);
    }
    return P{};
}

template<typename P, typename T>
[[nodiscard]] __device__ inline auto lc_texel_write_convert(T t) noexcept {
    if constexpr (lc_is_same<T, lc_float>::value()) {
        return lc_float_to_texel<P>(t);
    } else if constexpr (lc_is_same<T, lc_int>::value()) {
        return lc_int_to_texel<P>(t);
    } else if constexpr (lc_is_same<T, lc_uint>::value()) {
        return lc_uint_to_texel<P>(t);
    } else {
        static_assert(lc_always_false<T, P>::value());
    }
}

template<typename T>
struct lc_vec4 {};

template<>
struct lc_vec4<lc_int> {
    using type = lc_int4;
};

template<>
struct lc_vec4<lc_uint> {
    using type = lc_uint4;
};

template<>
struct lc_vec4<lc_float> {
    using type = lc_float4;
};

template<typename T>
using lc_vec4_t = typename lc_vec4<T>::type;

template<typename T>
[[nodiscard]] __device__ inline auto lc_surf2d_read(LCSurface surf, lc_uint2 p) noexcept {
    lc_vec4_t<T> result{0, 0, 0, 0};
    switch (static_cast<LCPixelStorage>(surf.storage)) {
        case LCPixelStorage::BYTE1: {
            int x;
            asm("suld.b.2d.b8.zero %0, [%1, {%2, %3}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(char)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, char>(x);
            break;
        }
        case LCPixelStorage::BYTE2: {
            int x, y;
            asm("suld.b.2d.v2.b8.zero {%0, %1}, [%2, {%3, %4}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(char2)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, char>(x);
            result.y = lc_texel_read_convert<T, char>(y);
            break;
        }
        case LCPixelStorage::BYTE4: {
            int x, y, z, w;
            asm("suld.b.2d.v4.b8.zero {%0, %1, %2, %3}, [%4, {%5, %6}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(char4)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, char>(x);
            result.y = lc_texel_read_convert<T, char>(y);
            result.z = lc_texel_read_convert<T, char>(z);
            result.w = lc_texel_read_convert<T, char>(w);
            break;
        }
        case LCPixelStorage::SHORT1: {
            int x;
            asm("suld.b.2d.b16.zero %0, [%1, {%2, %3}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(short)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, short>(x);
            break;
        }
        case LCPixelStorage::SHORT2: {
            int x, y;
            asm("suld.b.2d.v2.b16.zero {%0, %1}, [%2, {%3, %4}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(short2)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, short>(x);
            result.y = lc_texel_read_convert<T, short>(y);
            break;
        }
        case LCPixelStorage::SHORT4: {
            int x, y, z, w;
            asm("suld.b.2d.v4.b16.zero {%0, %1, %2, %3}, [%4, {%5, %6}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(short4)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, short>(x);
            result.y = lc_texel_read_convert<T, short>(y);
            result.z = lc_texel_read_convert<T, short>(z);
            result.w = lc_texel_read_convert<T, short>(w);
            break;
        }
        case LCPixelStorage::INT1: {
            int x;
            asm("suld.b.2d.b32.zero %0, [%1, {%2, %3}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(int)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, int>(x);
            break;
        }
        case LCPixelStorage::INT2: {
            int x, y;
            asm("suld.b.2d.v2.b32.zero {%0, %1}, [%2, {%3, %4}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(int2)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, int>(x);
            result.y = lc_texel_read_convert<T, int>(y);
            break;
        }
        case LCPixelStorage::INT4: {
            int x, y, z, w;
            asm("suld.b.2d.v4.b32.zero {%0, %1, %2, %3}, [%4, {%5, %6}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(int4)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, int>(x);
            result.y = lc_texel_read_convert<T, int>(y);
            result.z = lc_texel_read_convert<T, int>(z);
            result.w = lc_texel_read_convert<T, int>(w);
            break;
        }
        case LCPixelStorage::HALF1: {
            lc_uint x;
            asm("suld.b.2d.b16.zero %0, [%1, {%2, %3}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(lc_half)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, lc_half>(lc_half{static_cast<lc_ushort>(x)});
            break;
        }
        case LCPixelStorage::HALF2: {
            lc_uint x, y;
            asm("suld.b.2d.v2.b16.zero {%0, %1}, [%2, {%3, %4}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(lc_half2)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, lc_half>(lc_half{static_cast<lc_ushort>(x)});
            result.y = lc_texel_read_convert<T, lc_half>(lc_half{static_cast<lc_ushort>(y)});
            break;
        }
        case LCPixelStorage::HALF4: {
            lc_uint x, y, z, w;
            asm("suld.b.2d.v4.b16.zero {%0, %1, %2, %3}, [%4, {%5, %6}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(lc_half4)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, lc_half>(lc_half{static_cast<lc_ushort>(x)});
            result.y = lc_texel_read_convert<T, lc_half>(lc_half{static_cast<lc_ushort>(y)});
            result.z = lc_texel_read_convert<T, lc_half>(lc_half{static_cast<lc_ushort>(z)});
            result.w = lc_texel_read_convert<T, lc_half>(lc_half{static_cast<lc_ushort>(w)});
            break;
        }
        case LCPixelStorage::FLOAT1: {
            float x;
            asm("suld.b.2d.b32.zero %0, [%1, {%2, %3}];"
                : "=f"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(float)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, float>(x);
            break;
        }
        case LCPixelStorage::FLOAT2: {
            float x, y;
            asm("suld.b.2d.v2.b32.zero {%0, %1}, [%2, {%3, %4}];"
                : "=f"(x), "=f"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(float2)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, float>(x);
            result.y = lc_texel_read_convert<T, float>(y);
            break;
        }
        case LCPixelStorage::FLOAT4: {
            float x, y, z, w;
            asm("suld.b.2d.v4.b32.zero {%0, %1, %2, %3}, [%4, {%5, %6}];"
                : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(float4)), "r"(p.y)
                : "memory");
            result.x = lc_texel_read_convert<T, float>(x);
            result.y = lc_texel_read_convert<T, float>(y);
            result.z = lc_texel_read_convert<T, float>(z);
            result.w = lc_texel_read_convert<T, float>(w);
            break;
        }
        default: __builtin_unreachable();
    }
    return result;
}

template<typename T, typename V>
__device__ inline void lc_surf2d_write(LCSurface surf, lc_uint2 p, V value) noexcept {
    switch (static_cast<LCPixelStorage>(surf.storage)) {
        case LCPixelStorage::BYTE1: {
            int v = lc_texel_write_convert<char>(value.x);
            asm volatile("sust.b.2d.b8.zero [%0, {%1, %2}], %3;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(char))), "r"(p.y), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::BYTE2: {
            int vx = lc_texel_write_convert<char>(value.x);
            int vy = lc_texel_write_convert<char>(value.y);
            asm volatile("sust.b.2d.v2.b8.zero [%0, {%1, %2}], {%3, %4};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(char2))), "r"(p.y), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::BYTE4: {
            int vx = lc_texel_write_convert<char>(value.x);
            int vy = lc_texel_write_convert<char>(value.y);
            int vz = lc_texel_write_convert<char>(value.z);
            int vw = lc_texel_write_convert<char>(value.w);
            asm volatile("sust.b.2d.v4.b8.zero [%0, {%1, %2}], {%3, %4, %5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(char4))), "r"(p.y), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::SHORT1: {
            int v = lc_texel_write_convert<short>(value.x);
            asm volatile("sust.b.2d.b16.zero [%0, {%1, %2}], %3;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(short))), "r"(p.y), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::SHORT2: {
            int vx = lc_texel_write_convert<short>(value.x);
            int vy = lc_texel_write_convert<short>(value.y);
            asm volatile("sust.b.2d.v2.b16.zero [%0, {%1, %2}], {%3, %4};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(short2))), "r"(p.y), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::SHORT4: {
            int vx = lc_texel_write_convert<short>(value.x);
            int vy = lc_texel_write_convert<short>(value.y);
            int vz = lc_texel_write_convert<short>(value.z);
            int vw = lc_texel_write_convert<short>(value.w);
            asm volatile("sust.b.2d.v4.b16.zero [%0, {%1, %2}], {%3, %4, %5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(short4))), "r"(p.y), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::INT1: {
            int v = lc_texel_write_convert<int>(value.x);
            asm volatile("sust.b.2d.b32.zero [%0, {%1, %2}], %3;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(int))), "r"(p.y), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::INT2: {
            int vx = lc_texel_write_convert<int>(value.x);
            int vy = lc_texel_write_convert<int>(value.y);
            asm volatile("sust.b.2d.v2.b32.zero [%0, {%1, %2}], {%3, %4};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(int2))), "r"(p.y), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::INT4: {
            int vx = lc_texel_write_convert<int>(value.x);
            int vy = lc_texel_write_convert<int>(value.y);
            int vz = lc_texel_write_convert<int>(value.z);
            int vw = lc_texel_write_convert<int>(value.w);
            asm volatile("sust.b.2d.v4.b32.zero [%0, {%1, %2}], {%3, %4, %5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(int4))), "r"(p.y), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::HALF1: {
            lc_uint v = lc_texel_write_convert<lc_half>(value.x).bits;
            asm volatile("sust.b.2d.b16.zero [%0, {%1, %2}], %3;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(lc_half))), "r"(p.y), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::HALF2: {
            lc_uint vx = lc_texel_write_convert<lc_half>(value.x).bits;
            lc_uint vy = lc_texel_write_convert<lc_half>(value.y).bits;
            asm volatile("sust.b.2d.v2.b16.zero [%0, {%1, %2}], {%3, %4};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(lc_half2))), "r"(p.y), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::HALF4: {
            lc_uint vx = lc_texel_write_convert<lc_half>(value.x).bits;
            lc_uint vy = lc_texel_write_convert<lc_half>(value.y).bits;
            lc_uint vz = lc_texel_write_convert<lc_half>(value.z).bits;
            lc_uint vw = lc_texel_write_convert<lc_half>(value.w).bits;
            asm volatile("sust.b.2d.v4.b16.zero [%0, {%1, %2}], {%3, %4, %5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(lc_half4))), "r"(p.y), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::FLOAT1: {
            float v = lc_texel_write_convert<float>(value.x);
            asm volatile("sust.b.2d.b32.zero [%0, {%1, %2}], %3;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(float))), "r"(p.y), "f"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::FLOAT2: {
            float vx = lc_texel_write_convert<float>(value.x);
            float vy = lc_texel_write_convert<float>(value.y);
            asm volatile("sust.b.2d.v2.b32.zero [%0, {%1, %2}], {%3, %4};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(float2))), "r"(p.y), "f"(vx), "f"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::FLOAT4: {
            float vx = lc_texel_write_convert<float>(value.x);
            float vy = lc_texel_write_convert<float>(value.y);
            float vz = lc_texel_write_convert<float>(value.z);
            float vw = lc_texel_write_convert<float>(value.w);
            asm volatile("sust.b.2d.v4.b32.zero [%0, {%1, %2}], {%3, %4, %5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(float4))), "r"(p.y), "f"(vx), "f"(vy), "f"(vz), "f"(vw)
                         : "memory");
            break;
        }
        default: __builtin_unreachable();
    }
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_surf3d_read(LCSurface surf, lc_uint3 p) noexcept {
    lc_vec4_t<T> result{0, 0, 0, 0};
    switch (static_cast<LCPixelStorage>(surf.storage)) {
        case LCPixelStorage::BYTE1: {
            int x;
            asm("suld.b.3d.b8.zero %0, [%1, {%2, %3, %4, %5}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(char)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, char>(x);
            break;
        }
        case LCPixelStorage::BYTE2: {
            int x, y;
            asm("suld.b.3d.v2.b8.zero {%0, %1}, [%2, {%3, %4, %5, %6}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(char2)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert < T, char(x);
            result.y = lc_texel_read_convert < T, char(y);
            break;
        }
        case LCPixelStorage::BYTE4: {
            int x, y, z, w;
            asm("suld.b.3d.v4.b8.zero {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(char4)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, char>(x);
            result.y = lc_texel_read_convert<T, char>(y);
            result.z = lc_texel_read_convert<T, char>(z);
            result.w = lc_texel_read_convert<T, char>(w);
            break;
        }
        case LCPixelStorage::SHORT1: {
            int x;
            asm("suld.b.3d.b16.zero %0, [%1, {%2, %3, %4, %5}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(short)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, short>(x);
            break;
        }
        case LCPixelStorage::SHORT2: {
            int x, y;
            asm("suld.b.3d.v2.b16.zero {%0, %1}, [%2, {%3, %4, %5, %6}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(short2)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, short>(x);
            result.y = lc_texel_read_convert<T, short>(y);
            break;
        }
        case LCPixelStorage::SHORT4: {
            int x, y, z, w;
            asm("suld.b.3d.v4.b16.zero {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(short4)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, short>(x);
            result.y = lc_texel_read_convert<T, short>(y);
            result.z = lc_texel_read_convert<T, short>(z);
            result.w = lc_texel_read_convert<T, short>(w);
            break;
        }
        case LCPixelStorage::INT1: {
            int x;
            asm("suld.b.3d.b32.zero %0, [%1, {%2, %3, %4, %5}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(int)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, int>(x);
            break;
        }
        case LCPixelStorage::INT2: {
            int x, y;
            asm("suld.b.3d.v2.b32.zero {%0, %1}, [%2, {%3, %4, %5, %6}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(int2)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, int>(x);
            result.y = lc_texel_read_convert<T, int>(y);
            break;
        }
        case LCPixelStorage::INT4: {
            int x, y, z, w;
            asm("suld.b.3d.v4.b32.zero {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(int4)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, int>(x);
            result.y = lc_texel_read_convert<T, int>(y);
            result.z = lc_texel_read_convert<T, int>(z);
            result.w = lc_texel_read_convert<T, int>(w);
            break;
        }
        case LCPixelStorage::HALF1: {
            lc_uint x;
            asm("suld.b.3d.b16.zero %0, [%1, {%2, %3, %4, %5}];"
                : "=r"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(lc_half)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, lc_half>(x);
            break;
        }
        case LCPixelStorage::HALF2: {
            lc_uint x, y;
            asm("suld.b.3d.v2.b16.zero {%0, %1}, [%2, {%3, %4, %5, %6}];"
                : "=r"(x), "=r"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(lc_half2)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, lc_half>(x);
            result.y = lc_texel_read_convert<T, lc_half>(y);
            break;
        }
        case LCPixelStorage::HALF4: {
            lc_uint x, y, z, w;
            asm("suld.b.3d.v4.b16.zero {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
                : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(lc_half4)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, lc_half>(x);
            result.y = lc_texel_read_convert<T, lc_half>(y);
            result.z = lc_texel_read_convert<T, lc_half>(z);
            result.w = lc_texel_read_convert<T, lc_half>(w);
            break;
        }
        case LCPixelStorage::FLOAT1: {
            float x;
            asm("suld.b.3d.b32.zero %0, [%1, {%2, %3, %4, %5}];"
                : "=f"(x)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(float)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, float>(x);
            break;
        }
        case LCPixelStorage::FLOAT2: {
            float x, y;
            asm("suld.b.3d.v2.b32.zero {%0, %1}, [%2, {%3, %4, %5, %6}];"
                : "=f"(x), "=f"(y)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(float2)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, float>(x);
            result.y = lc_texel_read_convert<T, float>(y);
            break;
        }
        case LCPixelStorage::FLOAT4: {
            float x, y, z, w;
            asm("suld.b.3d.v4.b32.zero {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
                : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
                : "l"(surf.handle), "r"(p.x * (int)sizeof(float4)), "r"(p.y), "r"(p.z), "r"(0)
                : "memory");
            result.x = lc_texel_read_convert<T, float>(x);
            result.y = lc_texel_read_convert<T, float>(y);
            result.z = lc_texel_read_convert<T, float>(z);
            result.w = lc_texel_read_convert<T, float>(w);
            break;
        }
        default: __builtin_unreachable();
    }
    return result;
}

template<typename T, typename V>
__device__ inline void lc_surf3d_write(LCSurface surf, lc_uint3 p, V value) noexcept {
    switch (static_cast<LCPixelStorage>(surf.storage)) {
        case LCPixelStorage::BYTE1: {
            int v = lc_texel_write_convert<char>(value.x);
            asm volatile("sust.b.3d.b8.zero [%0, {%1, %2, %3, %4}], %5;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(char))), "r"(p.y), "r"(p.z), "r"(0), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::BYTE2: {
            int vx = lc_texel_write_convert<char>(value.x);
            int vy = lc_texel_write_convert<char>(value.y);
            asm volatile("sust.b.3d.v2.b8.zero [%0, {%1, %2, %3, %4}], {%5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(char2))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::BYTE4: {
            int vx = lc_texel_write_convert<char>(value.x);
            int vy = lc_texel_write_convert<char>(value.y);
            int vz = lc_texel_write_convert<char>(value.z);
            int vw = lc_texel_write_convert<char>(value.w);
            asm volatile("sust.b.3d.v4.b8.zero [%0, {%1, %2, %3, %4}], {%5, %6, %7, %8};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(char4))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::SHORT1: {
            int v = lc_texel_write_convert<short>(value.x);
            asm volatile("sust.b.3d.b16.zero [%0, {%1, %2, %3, %4}], %5;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(short))), "r"(p.y), "r"(p.z), "r"(0), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::SHORT2: {
            int vx = lc_texel_write_convert<short>(value.x);
            int vy = lc_texel_write_convert<short>(value.y);
            asm volatile("sust.b.3d.v2.b16.zero [%0, {%1, %2, %3, %4}], {%5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(short2))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::SHORT4: {
            int vx = lc_texel_write_convert<short>(value.x);
            int vy = lc_texel_write_convert<short>(value.y);
            int vz = lc_texel_write_convert<short>(value.z);
            int vw = lc_texel_write_convert<short>(value.w);
            asm volatile("sust.b.3d.v4.b16.zero [%0, {%1, %2, %3, %4}], {%5, %6, %7, %8};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(short4))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::INT1: {
            int v = lc_texel_write_convert<int>(value.x);
            asm volatile("sust.b.3d.b32.zero [%0, {%1, %2, %3, %4}], %5;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(int))), "r"(p.y), "r"(p.z), "r"(0), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::INT2: {
            int vx = lc_texel_write_convert<int>(value.x);
            int vy = lc_texel_write_convert<int>(value.y);
            asm volatile("sust.b.3d.v2.b32.zero [%0, {%1, %2, %3, %4}], {%5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(int2))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::INT4: {
            int vx = lc_texel_write_convert<int>(value.x);
            int vy = lc_texel_write_convert<int>(value.y);
            int vz = lc_texel_write_convert<int>(value.z);
            int vw = lc_texel_write_convert<int>(value.w);
            asm volatile("sust.b.3d.v4.b32.zero [%0, {%1, %2, %3, %4}], {%5, %6, %7, %8};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(int4))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::HALF1: {
            lc_uint v = lc_texel_write_convert<lc_half>(value.x).bits;
            asm volatile("sust.b.3d.b16.zero [%0, {%1, %2, %3, %4}], %5;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(lc_half))), "r"(p.y), "r"(p.z), "r"(0), "r"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::HALF2: {
            lc_uint vx = lc_texel_write_convert<lc_half>(value.x).bits;
            lc_uint vy = lc_texel_write_convert<lc_half>(value.y).bits;
            asm volatile("sust.b.3d.v2.b16.zero [%0, {%1, %2, %3, %4}], {%5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(short2))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::HALF4: {
            lc_uint vx = lc_texel_write_convert<lc_half>(value.x).bits;
            lc_uint vy = lc_texel_write_convert<lc_half>(value.y).bits;
            lc_uint vz = lc_texel_write_convert<lc_half>(value.z).bits;
            lc_uint vw = lc_texel_write_convert<lc_half>(value.w).bits;
            asm volatile("sust.b.3d.v4.b16.zero [%0, {%1, %2, %3, %4}], {%5, %6, %7, %8};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(lc_half4))), "r"(p.y), "r"(p.z), "r"(0), "r"(vx), "r"(vy), "r"(vz), "r"(vw)
                         : "memory");
            break;
        }
        case LCPixelStorage::FLOAT1: {
            float v = lc_texel_write_convert<float>(value.x);
            asm volatile("sust.b.3d.b32.zero [%0, {%1, %2, %3, %4}], %5;"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(float))), "r"(p.y), "r"(p.z), "r"(0), "f"(v)
                         : "memory");
            break;
        }
        case LCPixelStorage::FLOAT2: {
            float vx = lc_texel_write_convert<float>(value.x);
            float vy = lc_texel_write_convert<float>(value.y);
            asm volatile("sust.b.3d.v2.b32.zero [%0, {%1, %2, %3, %4}], {%5, %6};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(float2))), "r"(p.y), "r"(p.z), "r"(0), "f"(vx), "f"(vy)
                         : "memory");
            break;
        }
        case LCPixelStorage::FLOAT4: {
            float vx = lc_texel_write_convert<float>(value.x);
            float vy = lc_texel_write_convert<float>(value.y);
            float vz = lc_texel_write_convert<float>(value.z);
            float vw = lc_texel_write_convert<float>(value.w);
            asm volatile("sust.b.3d.v4.b32.zero [%0, {%1, %2, %3, %4}], {%5, %6, %7, %8};"
                         :
                         : "l"(surf.handle), "r"(p.x * (int)(sizeof(float4))), "r"(p.y), "r"(p.z), "r"(0), "f"(vx), "f"(vy), "f"(vz), "f"(vw)
                         : "memory");
            break;
        }
        default: __builtin_unreachable();
    }
}

template<typename T>
struct LCTexture2D {
    LCSurface surface;
};

template<typename T>
struct LCTexture3D {
    LCSurface surface;
};

template<typename T>
[[nodiscard]] __device__ inline auto lc_texture_size(LCTexture2D<T> tex) noexcept {
    lc_uint2 size;
    asm("suq.width.b32 %0, [%1];"
        : "=r"(size.x)
        : "l"(tex.surface.handle));
    asm("suq.height.b32 %0, [%1];"
        : "=r"(size.y)
        : "l"(tex.surface.handle));
    return size;
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_texture_size(LCTexture3D<T> tex) noexcept {
    lc_uint3 size;
    asm("suq.width.b32 %0, [%1];"
        : "=r"(size.x)
        : "l"(tex.surface.handle));
    asm("suq.height.b32 %0, [%1];"
        : "=r"(size.y)
        : "l"(tex.surface.handle));
    asm("suq.depth.b32 %0, [%1];"
        : "=r"(size.z)
        : "l"(tex.surface.handle));
    return size;
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_texture_read(LCTexture2D<T> tex, lc_uint2 p) noexcept {
    return lc_surf2d_read<T>(tex.surface, p);
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_texture_read(LCTexture3D<T> tex, lc_uint3 p) noexcept {
    return lc_surf3d_read<T>(tex.surface, p);
}

template<typename T, typename V>
__device__ inline void lc_texture_write(LCTexture2D<T> tex, lc_uint2 p, V value) noexcept {
    lc_surf2d_write<T>(tex.surface, p, value);
}

template<typename T, typename V>
__device__ inline void lc_texture_write(LCTexture3D<T> tex, lc_uint3 p, V value) noexcept {
    lc_surf3d_write<T>(tex.surface, p, value);
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_texture_read(LCTexture2D<T> tex, lc_int2 p) noexcept {
    return lc_texture_read(tex, lc_make_uint2(p));
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_texture_read(LCTexture3D<T> tex, lc_int3 p) noexcept {
    return lc_texture_read(tex, lc_make_uint3(p));
}

template<typename T, typename V>
__device__ inline void lc_texture_write(LCTexture2D<T> tex, lc_int2 p, V value) noexcept {
    lc_texture_write(tex, lc_make_uint2(p), value);
}

template<typename T, typename V>
__device__ inline void lc_texture_write(LCTexture3D<T> tex, lc_int3 p, V value) noexcept {
    lc_texture_write(tex, lc_make_uint3(p), value);
}

struct alignas(16) LCBindlessSlot {
    const void *__restrict__ buffer;
    size_t buffer_size;
    cudaTextureObject_t tex2d;
    cudaTextureObject_t tex3d;
};

struct alignas(16) LCBindlessArray {
    const LCBindlessSlot *__restrict__ slots;
};

template<typename T = unsigned char>
[[nodiscard]] inline __device__ auto lc_bindless_buffer_size(LCBindlessArray array, lc_uint index) noexcept {
    lc_assume(__isGlobal(array.slots));
    return array.slots[index].buffer_size / sizeof(T);
}

[[nodiscard]] inline __device__ auto lc_bindless_buffer_size(LCBindlessArray array, lc_uint index, lc_uint stride) noexcept {
    lc_assume(__isGlobal(array.slots));
    return array.slots[index].buffer_size / stride;
}

template<typename T>
[[nodiscard]] inline __device__ auto lc_bindless_buffer_read(LCBindlessArray array, lc_uint index, lc_ulong i) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto buffer = static_cast<const T *>(array.slots[index].buffer);
    lc_assume(__isGlobal(buffer));
#ifdef LUISA_DEBUG
    lc_check_in_bounds(i, lc_bindless_buffer_size<T>(array, index));
#endif
    return buffer[i];
}

[[nodiscard]] inline __device__ auto lc_bindless_buffer_type(LCBindlessArray array, lc_uint index) noexcept {
    return 0ull;// TODO
}

template<typename T>
[[nodiscard]] inline __device__ auto lc_bindless_byte_buffer_read(LCBindlessArray array, lc_uint index, lc_ulong offset) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto buffer = static_cast<const char *>(array.slots[index].buffer);
    lc_assume(__isGlobal(buffer));
#ifdef LUISA_DEBUG
    lc_check_in_bounds(offset + sizeof(T), lc_bindless_buffer_size<char>(array, index));
#endif
    return *reinterpret_cast<const T *>(buffer + offset);
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample2d(LCBindlessArray array, lc_uint index, lc_float2 p) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex2d;
    auto v = lc_make_float4();
    asm("tex.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "f"(p.x), "f"(p.y));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample3d(LCBindlessArray array, lc_uint index, lc_float3 p) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex3d;
    auto v = lc_make_float4();
    asm("tex.3d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "f"(p.x), "f"(p.y), "f"(p.z), "f"(0.f));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample2d_level(LCBindlessArray array, lc_uint index, lc_float2 p, float level) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex2d;
    auto v = lc_make_float4();
    asm("tex.level.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "f"(p.x), "f"(p.y), "f"(level));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample3d_level(LCBindlessArray array, lc_uint index, lc_float3 p, float level) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex3d;
    auto v = lc_make_float4();
    asm("tex.3d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}], %9;"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "f"(p.x), "f"(p.y), "f"(p.z), "f"(0.f), "f"(level));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample2d_grad(LCBindlessArray array, lc_uint index, lc_float2 p, lc_float2 dx, lc_float2 dy) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex2d;
    auto v = lc_make_float4();
    asm("tex.grad.2d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6}], {%7, %8}, {%9, %10};"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "f"(p.x), "f"(p.y), "f"(dx.x), "f"(dx.y), "f"(dy.x), "f"(dy.y));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_sample3d_grad(LCBindlessArray array, lc_uint index, lc_float3 p, lc_float3 dx, lc_float3 dy) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex3d;
    auto v = lc_make_float4();
    asm("tex.grad.3d.v4.f32.f32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}], {%9, %10, %11, %12}, {%13, %14, %15, 16};"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "f"(p.x), "f"(p.y), "f"(p.z), "f"(0.f),
          "f"(dx.x), "f"(dx.y), "f"(dx.z), "f"(0.f),
          "f"(dy.x), "f"(dy.y), "f"(dy.z), "f"(0.f));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_size2d(LCBindlessArray array, lc_uint index) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex2d;
    auto s = lc_make_uint2();
    asm("txq.width.b32 %0, [%1];"
        : "=r"(s.x)
        : "l"(t));
    asm("txq.height.b32 %0, [%1];"
        : "=r"(s.y)
        : "l"(t));
    return s;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_size3d(LCBindlessArray array, lc_uint index) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex3d;
    auto s = lc_make_uint3();
    asm("txq.width.b32 %0, [%1];"
        : "=r"(s.x)
        : "l"(t));
    asm("txq.height.b32 %0, [%1];"
        : "=r"(s.y)
        : "l"(t));
    asm("txq.depth.b32 %0, [%1];"
        : "=r"(s.z)
        : "l"(t));
    return s;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_size2d_level(LCBindlessArray array, lc_uint index, lc_uint level) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto s = lc_bindless_texture_size2d(array, index);
    return lc_max(s >> level, lc_make_uint2(1u));
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_size3d_level(LCBindlessArray array, lc_uint index, lc_uint level) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto s = lc_bindless_texture_size3d(array, index);
    return lc_max(s >> level, lc_make_uint3(1u));
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_read2d(LCBindlessArray array, lc_uint index, lc_uint2 p) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex2d;
    auto v = lc_make_float4();
    asm("tex.2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "r"(p.x), "r"(p.y));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_read3d(LCBindlessArray array, lc_uint index, lc_uint3 p) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex3d;
    auto v = lc_make_float4();
    asm("tex.3d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}];"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "r"(p.x), "r"(p.y), "r"(p.z), "r"(0u));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_read2d_level(LCBindlessArray array, lc_uint index, lc_uint2 p, lc_uint level) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex2d;
    auto v = lc_make_float4();
    asm("tex.level.2d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6}], %7;"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "r"(p.x), "r"(p.y), "r"(level));
    return v;
}

[[nodiscard]] inline __device__ auto lc_bindless_texture_read3d_level(LCBindlessArray array, lc_uint index, lc_uint3 p, lc_uint level) noexcept {
    lc_assume(__isGlobal(array.slots));
    auto t = array.slots[index].tex3d;
    auto v = lc_make_float4();
    asm("tex.level.3d.v4.f32.s32 {%0, %1, %2, %3}, [%4, {%5, %6, %7, %8}], %9;"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(t), "r"(p.x), "r"(p.y), "r"(p.z), "r"(0u), "r"(level));
    return v;
}

__device__ inline float atomicCAS(float *a, float cmp, float v) noexcept {
    return __uint_as_float(atomicCAS(reinterpret_cast<lc_uint *>(a),
                                     __float_as_uint(cmp),
                                     __float_as_uint(v)));
}

__device__ inline float atomicSub(float *a, float v) noexcept {
    return atomicAdd(a, -v);
}

__device__ inline float atomicMin(float *a, float v) noexcept {
    for (;;) {
        if (auto old = *a;// read old
            old <= v /* no need to update */ ||
            atomicCAS(a, old, v) == old) { return old; }
    }
}

__device__ inline float atomicMax(float *a, float v) noexcept {
    for (;;) {
        if (auto old = *a;// read old
            old >= v /* no need to update */ ||
            atomicCAS(a, old, v) == old) { return old; }
    }
}

#define lc_atomic_exchange(atomic_ref, value) atomicExch(&(atomic_ref), value)
#define lc_atomic_compare_exchange(atomic_ref, cmp, value) atomicCAS(&(atomic_ref), cmp, value)
#define lc_atomic_fetch_add(atomic_ref, value) atomicAdd(&(atomic_ref), value)
#define lc_atomic_fetch_sub(atomic_ref, value) atomicSub(&(atomic_ref), value)
#define lc_atomic_fetch_min(atomic_ref, value) atomicMin(&(atomic_ref), value)
#define lc_atomic_fetch_max(atomic_ref, value) atomicMax(&(atomic_ref), value)
#define lc_atomic_fetch_and(atomic_ref, value) atomicAnd(&(atomic_ref), value)
#define lc_atomic_fetch_or(atomic_ref, value) atomicOr(&(atomic_ref), value)
#define lc_atomic_fetch_xor(atomic_ref, value) atomicXor(&(atomic_ref), value)

// static block size
[[nodiscard]] __device__ constexpr lc_uint3 lc_block_size() noexcept {
    return LC_BLOCK_SIZE;
}

#define lc_dispatch_size() lc_make_uint3(params.ls_kid)
#define lc_kernel_id() static_cast<lc_uint>(params.ls_kid.w)

inline void lc_shader_execution_reorder(lc_uint hint, lc_uint hint_bits) noexcept {
    // do nothing since SER is not supported in plain CUDA
}

[[nodiscard]] __device__ inline auto lc_thread_id() noexcept {
    return lc_make_uint3(lc_uint(threadIdx.x),
                         lc_uint(threadIdx.y),
                         lc_uint(threadIdx.z));
}

[[nodiscard]] __device__ inline auto lc_block_id() noexcept {
    return lc_make_uint3(lc_uint(blockIdx.x),
                         lc_uint(blockIdx.y),
                         lc_uint(blockIdx.z));
}

[[nodiscard]] __device__ inline auto lc_dispatch_id() noexcept {
    return lc_block_id() * lc_block_size() + lc_thread_id();
}

__device__ inline void lc_synchronize_block() noexcept {
    __syncthreads();
}

// autodiff
#define LC_GRAD_SHADOW_VARIABLE(x) auto x##_grad = lc_zero<decltype(x)>()
#define LC_MARK_GRAD(x, dx) x##_grad = dx
#define LC_GRAD(x) (x##_grad)
#define LC_ACCUM_GRAD(x_grad, dx) lc_accumulate_grad(&(x_grad), (dx))
#define LC_REQUIRES_GRAD(x) x##_grad = lc_zero<decltype(x##_grad)>()

template<typename T>
struct alignas(alignof(T) < 4u ? 4u : alignof(T)) LCPack {
    T value;
};

template<typename T>
__device__ inline void lc_pack_to(const T &x, LCBuffer<lc_uint> array, lc_uint idx) noexcept {
    constexpr lc_uint N = (sizeof(T) + 3u) / 4u;
    if constexpr (alignof(T) < 4u) {
        // too small to be aligned to 4 bytes
        LCPack<T> pack{};
        pack.value = x;
        auto data = reinterpret_cast<const lc_uint *>(&pack);
#pragma unroll
        for (auto i = 0u; i < N; i++) {
            array.ptr[idx + i] = data[i];
        }
    } else {
        // safe to reinterpret the pointer as lc_uint *
        auto data = reinterpret_cast<const lc_uint *>(&x);
#pragma unroll
        for (auto i = 0u; i < N; i++) {
            array.ptr[idx + i] = data[i];
        }
    }
}

template<typename T>
[[nodiscard]] __device__ inline T lc_unpack_from(LCBuffer<lc_uint> array, lc_uint idx) noexcept {
    if constexpr (alignof(T) <= 4u) {
        // safe to reinterpret the pointer as T *
        auto data = reinterpret_cast<const T *>(&array.ptr[idx]);
        return *data;
    } else {
        // copy to a temporary aligned buffer to avoid unaligned access
        constexpr lc_uint N = (sizeof(T) + 3u) / 4u;
        LCPack<T> x{};
        auto data = reinterpret_cast<lc_uint *>(&x);
#pragma unroll
        for (auto i = 0u; i < N; i++) {
            data[i] = array.ptr[idx + i];
        }
        return x.value;
    }
}

using lc_byte = unsigned char;

template<typename T>
[[nodiscard]] __device__ inline T lc_byte_buffer_read(LCBuffer<const lc_byte> buffer, lc_ulong offset) noexcept {
    lc_assume(__isGlobal(buffer.ptr));
    auto address = reinterpret_cast<lc_ulong>(buffer.ptr + offset);
#ifdef LUISA_DEBUG
    lc_check_in_bounds(offset + sizeof(T), lc_buffer_size(buffer));
    lc_assert(address % alignof(T) == 0u && "unaligned access");
#endif
    return *reinterpret_cast<T *>(address);
}

template<typename T>
__device__ inline void lc_byte_buffer_write(LCBuffer<lc_byte> buffer, lc_ulong offset, T value) noexcept {
    lc_assume(__isGlobal(buffer.ptr));
    auto address = reinterpret_cast<lc_ulong>(buffer.ptr + offset);
#ifdef LUISA_DEBUG
    lc_check_in_bounds(offset + sizeof(T), lc_buffer_size(buffer));
    lc_assert(address % alignof(T) == 0u && "unaligned access");
#endif
    *reinterpret_cast<T *>(address) = value;
}

[[nodiscard]] __device__ inline auto lc_byte_buffer_size(LCBuffer<const lc_byte> buffer) noexcept {
    return lc_buffer_size(buffer);
}

// warp intrinsics
[[nodiscard]] __device__ inline auto lc_warp_lane_id() noexcept {
    lc_uint ret;
    asm("mov.u32 %0, %laneid;"
        : "=r"(ret));
    return ret;
}

[[nodiscard]] __device__ constexpr auto lc_warp_size() noexcept {
    return static_cast<lc_uint>(warpSize);
}

#define LC_WARP_FULL_MASK 0xffff'ffffu
#define LC_WARP_ACTIVE_MASK __activemask()

[[nodiscard]] __device__ inline auto lc_warp_first_active_lane() noexcept {
    return __ffs(LC_WARP_ACTIVE_MASK) - 1u;
}

[[nodiscard]] __device__ inline auto lc_warp_is_first_active_lane() noexcept {
    return lc_warp_first_active_lane() == lc_warp_lane_id();
}

#if __CUDA_ARCH__ >= 700
#define LC_WARP_ALL_EQ_SCALAR(T)                                                  \
    [[nodiscard]] __device__ inline auto lc_warp_active_all_equal(T x) noexcept { \
        auto mask = LC_WARP_ACTIVE_MASK;                                          \
        auto pred = 0;                                                            \
        __match_all_sync(mask, x, &pred);                                         \
        return pred != 0;                                                         \
    }
#else
#define LC_WARP_ALL_EQ_SCALAR(T)                                                  \
    [[nodiscard]] __device__ inline auto lc_warp_active_all_equal(T x) noexcept { \
        auto mask = LC_WARP_ACTIVE_MASK;                                          \
        auto first = __ffs(mask) - 1u;                                            \
        auto x0 = __shfl_sync(mask, x, first);                                    \
        return static_cast<bool>(__all_sync(mask, x == x0));                      \
    }
#endif

#define LC_WARP_ALL_EQ_VECTOR2(T)                                                    \
    [[nodiscard]] __device__ inline auto lc_warp_active_all_equal(T##2 v) noexcept { \
        return lc_make_bool2(lc_warp_active_all_equal(v.x),                          \
                             lc_warp_active_all_equal(v.y));                         \
    }

#define LC_WARP_ALL_EQ_VECTOR3(T)                                                    \
    [[nodiscard]] __device__ inline auto lc_warp_active_all_equal(T##3 v) noexcept { \
        return lc_make_bool3(lc_warp_active_all_equal(v.x),                          \
                             lc_warp_active_all_equal(v.y),                          \
                             lc_warp_active_all_equal(v.z));                         \
    }

#define LC_WARP_ALL_EQ_VECTOR4(T)                                                    \
    [[nodiscard]] __device__ inline auto lc_warp_active_all_equal(T##4 v) noexcept { \
        return lc_make_bool4(lc_warp_active_all_equal(v.x),                          \
                             lc_warp_active_all_equal(v.y),                          \
                             lc_warp_active_all_equal(v.z),                          \
                             lc_warp_active_all_equal(v.w));                         \
    }

#define LC_WARP_ALL_EQ(T)     \
    LC_WARP_ALL_EQ_SCALAR(T)  \
    LC_WARP_ALL_EQ_VECTOR2(T) \
    LC_WARP_ALL_EQ_VECTOR3(T) \
    LC_WARP_ALL_EQ_VECTOR4(T)

LC_WARP_ALL_EQ(lc_bool)
LC_WARP_ALL_EQ(lc_short)
LC_WARP_ALL_EQ(lc_ushort)
LC_WARP_ALL_EQ(lc_int)
LC_WARP_ALL_EQ(lc_uint)
LC_WARP_ALL_EQ(lc_long)
LC_WARP_ALL_EQ(lc_ulong)
LC_WARP_ALL_EQ(lc_float)
//LC_WARP_ALL_EQ(lc_half)// TODO
//LC_WARP_ALL_EQ(lc_double)// TODO

#undef LC_WARP_ALL_EQ_SCALAR
#undef LC_WARP_ALL_EQ_VECTOR2
#undef LC_WARP_ALL_EQ_VECTOR3
#undef LC_WARP_ALL_EQ_VECTOR4
#undef LC_WARP_ALL_EQ

template<typename T, typename F>
[[nodiscard]] __device__ inline auto lc_warp_active_reduce_impl(T x, F f) noexcept {
    auto mask = LC_WARP_ACTIVE_MASK;
    auto lane = lc_warp_lane_id();
    if (auto y = __shfl_xor_sync(mask, x, 0x10u); mask & (1u << (lane ^ 0x10u))) { x = f(x, y); }
    if (auto y = __shfl_xor_sync(mask, x, 0x08u); mask & (1u << (lane ^ 0x08u))) { x = f(x, y); }
    if (auto y = __shfl_xor_sync(mask, x, 0x04u); mask & (1u << (lane ^ 0x04u))) { x = f(x, y); }
    if (auto y = __shfl_xor_sync(mask, x, 0x02u); mask & (1u << (lane ^ 0x02u))) { x = f(x, y); }
    if (auto y = __shfl_xor_sync(mask, x, 0x01u); mask & (1u << (lane ^ 0x01u))) { x = f(x, y); }
    return x;
}

template<typename T>
[[nodiscard]] __device__ constexpr T lc_bit_and(T x, T y) noexcept { return x & y; }

template<typename T>
[[nodiscard]] __device__ constexpr T lc_bit_or(T x, T y) noexcept { return x | y; }

template<typename T>
[[nodiscard]] __device__ constexpr T lc_bit_xor(T x, T y) noexcept { return x ^ y; }

#define LC_WARP_REDUCE_BIT_SCALAR_FALLBACK(op, T)                                     \
    [[nodiscard]] __device__ inline auto lc_warp_active_bit_##op(lc_##T x) noexcept { \
        return static_cast<lc_##T>(lc_warp_active_reduce_impl(                        \
            x, [](lc_##T a, lc_##T b) noexcept { return lc_bit_##op(a, b); }));       \
    }

#if __CUDA_ARCH__ >= 800
#define LC_WARP_REDUCE_BIT_SCALAR(op, T)                                              \
    [[nodiscard]] __device__ inline auto lc_warp_active_bit_##op(lc_##T x) noexcept { \
        return static_cast<lc_##T>(__reduce_##op##_sync(LC_WARP_ACTIVE_MASK,          \
                                                        static_cast<lc_uint>(x)));    \
    }
#else
#define LC_WARP_REDUCE_BIT_SCALAR(op, T) LC_WARP_REDUCE_BIT_SCALAR_FALLBACK(op, T)
#endif

LC_WARP_REDUCE_BIT_SCALAR(and, uint)
LC_WARP_REDUCE_BIT_SCALAR(or, uint)
LC_WARP_REDUCE_BIT_SCALAR(xor, uint)
LC_WARP_REDUCE_BIT_SCALAR(and, int)
LC_WARP_REDUCE_BIT_SCALAR(or, int)
LC_WARP_REDUCE_BIT_SCALAR(xor, int)

LC_WARP_REDUCE_BIT_SCALAR(and, ushort)
LC_WARP_REDUCE_BIT_SCALAR(or, ushort)
LC_WARP_REDUCE_BIT_SCALAR(xor, ushort)
LC_WARP_REDUCE_BIT_SCALAR(and, short)
LC_WARP_REDUCE_BIT_SCALAR(or, short)
LC_WARP_REDUCE_BIT_SCALAR(xor, short)

LC_WARP_REDUCE_BIT_SCALAR_FALLBACK(and, ulong)
LC_WARP_REDUCE_BIT_SCALAR_FALLBACK(or, ulong)
LC_WARP_REDUCE_BIT_SCALAR_FALLBACK(xor, ulong)
LC_WARP_REDUCE_BIT_SCALAR_FALLBACK(and, long)
LC_WARP_REDUCE_BIT_SCALAR_FALLBACK(or, long)
LC_WARP_REDUCE_BIT_SCALAR_FALLBACK(xor, long)

#undef LC_WARP_REDUCE_BIT_SCALAR_FALLBACK
#undef LC_WARP_REDUCE_BIT_SCALAR

#define LC_WARP_REDUCE_BIT_VECTOR(op, T)                                                 \
    [[nodiscard]] __device__ inline auto lc_warp_active_bit_##op(lc_##T##2 v) noexcept { \
        return lc_make_##T##2(lc_warp_active_bit_##op(v.x),                              \
                              lc_warp_active_bit_##op(v.y));                             \
    }                                                                                    \
    [[nodiscard]] __device__ inline auto lc_warp_active_bit_##op(lc_##T##3 v) noexcept { \
        return lc_make_##T##3(lc_warp_active_bit_##op(v.x),                              \
                              lc_warp_active_bit_##op(v.y),                              \
                              lc_warp_active_bit_##op(v.z));                             \
    }                                                                                    \
    [[nodiscard]] __device__ inline auto lc_warp_active_bit_##op(lc_##T##4 v) noexcept { \
        return lc_make_##T##4(lc_warp_active_bit_##op(v.x),                              \
                              lc_warp_active_bit_##op(v.y),                              \
                              lc_warp_active_bit_##op(v.z),                              \
                              lc_warp_active_bit_##op(v.w));                             \
    }

LC_WARP_REDUCE_BIT_VECTOR(and, uint)
LC_WARP_REDUCE_BIT_VECTOR(or, uint)
LC_WARP_REDUCE_BIT_VECTOR(xor, uint)
LC_WARP_REDUCE_BIT_VECTOR(and, int)
LC_WARP_REDUCE_BIT_VECTOR(or, int)
LC_WARP_REDUCE_BIT_VECTOR(xor, int)

#undef LC_WARP_REDUCE_BIT_VECTOR

[[nodiscard]] __device__ inline auto lc_warp_active_bit_mask(bool pred) noexcept {
    return lc_make_uint4(__ballot_sync(LC_WARP_ACTIVE_MASK, pred), 0u, 0u, 0u);
}

[[nodiscard]] __device__ inline auto lc_warp_active_count_bits(bool pred) noexcept {
    return lc_popcount(__ballot_sync(LC_WARP_ACTIVE_MASK, pred));
}

[[nodiscard]] __device__ inline auto lc_warp_active_all(bool pred) noexcept {
    return static_cast<lc_bool>(__all_sync(LC_WARP_ACTIVE_MASK, pred));
}

[[nodiscard]] __device__ inline auto lc_warp_active_any(bool pred) noexcept {
    return static_cast<lc_bool>(__any_sync(LC_WARP_ACTIVE_MASK, pred));
}

[[nodiscard]] __device__ inline auto lc_warp_prefix_mask() noexcept {
    lc_uint ret;
    asm("mov.u32 %0, %lanemask_lt;"
        : "=r"(ret));
    return ret;
}

[[nodiscard]] __device__ inline auto lc_warp_prefix_count_bits(bool pred) noexcept {
    return lc_popcount(__ballot_sync(LC_WARP_ACTIVE_MASK, pred) & lc_warp_prefix_mask());
}

#define LC_WARP_READ_LANE_SCALAR(T)                                                        \
    [[nodiscard]] __device__ inline auto lc_warp_read_lane(lc_##T x, lc_uint i) noexcept { \
        return static_cast<lc_##T>(__shfl_sync(LC_WARP_ACTIVE_MASK, x, i));                \
    }

#define LC_WARP_READ_LANE_VECTOR2(T)                                                          \
    [[nodiscard]] __device__ inline auto lc_warp_read_lane(lc_##T##2 v, lc_uint i) noexcept { \
        return lc_make_##T##2(lc_warp_read_lane(v.x, i),                                      \
                              lc_warp_read_lane(v.y, i));                                     \
    }

#define LC_WARP_READ_LANE_VECTOR3(T)                                                          \
    [[nodiscard]] __device__ inline auto lc_warp_read_lane(lc_##T##3 v, lc_uint i) noexcept { \
        return lc_make_##T##3(lc_warp_read_lane(v.x, i),                                      \
                              lc_warp_read_lane(v.y, i),                                      \
                              lc_warp_read_lane(v.z, i));                                     \
    }

#define LC_WARP_READ_LANE_VECTOR4(T)                                                          \
    [[nodiscard]] __device__ inline auto lc_warp_read_lane(lc_##T##4 v, lc_uint i) noexcept { \
        return lc_make_##T##4(lc_warp_read_lane(v.x, i),                                      \
                              lc_warp_read_lane(v.y, i),                                      \
                              lc_warp_read_lane(v.z, i),                                      \
                              lc_warp_read_lane(v.w, i));                                     \
    }

#define LC_WARP_READ_LANE(T)     \
    LC_WARP_READ_LANE_SCALAR(T)  \
    LC_WARP_READ_LANE_VECTOR2(T) \
    LC_WARP_READ_LANE_VECTOR3(T) \
    LC_WARP_READ_LANE_VECTOR4(T)

LC_WARP_READ_LANE(bool)
LC_WARP_READ_LANE(short)
LC_WARP_READ_LANE(ushort)
LC_WARP_READ_LANE(int)
LC_WARP_READ_LANE(uint)
LC_WARP_READ_LANE(long)
LC_WARP_READ_LANE(ulong)
LC_WARP_READ_LANE(float)
//LC_WARP_READ_LANE(half)// TODO
//LC_WARP_READ_LANE(double)// TODO

#undef LC_WARP_READ_LANE_SCALAR
#undef LC_WARP_READ_LANE_VECTOR2
#undef LC_WARP_READ_LANE_VECTOR3
#undef LC_WARP_READ_LANE_VECTOR4
#undef LC_WARP_READ_LANE

[[nodiscard]] __device__ inline auto lc_warp_read_lane(lc_float2x2 m, lc_uint i) noexcept {
    return lc_make_float2x2(lc_warp_read_lane(m[0], i),
                            lc_warp_read_lane(m[1], i));
}

[[nodiscard]] __device__ inline auto lc_warp_read_lane(lc_float3x3 m, lc_uint i) noexcept {
    return lc_make_float3x3(lc_warp_read_lane(m[0], i),
                            lc_warp_read_lane(m[1], i),
                            lc_warp_read_lane(m[2], i));
}

[[nodiscard]] __device__ inline auto lc_warp_read_lane(lc_float4x4 m, lc_uint i) noexcept {
    return lc_make_float4x4(lc_warp_read_lane(m[0], i),
                            lc_warp_read_lane(m[1], i),
                            lc_warp_read_lane(m[2], i),
                            lc_warp_read_lane(m[3], i));
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_warp_read_first_active_lane(T x) noexcept {
    return lc_warp_read_lane(x, lc_warp_first_active_lane());
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_warp_active_min_impl(T x) noexcept {
    return lc_warp_active_reduce_impl(x, [](T a, T b) noexcept { return min(a, b); });
}
template<typename T>
[[nodiscard]] __device__ inline auto lc_warp_active_max_impl(T x) noexcept {
    return lc_warp_active_reduce_impl(x, [](T a, T b) noexcept { return max(a, b); });
}
template<typename T>
[[nodiscard]] __device__ inline auto lc_warp_active_sum_impl(T x) noexcept {
    return lc_warp_active_reduce_impl(x, [](T a, T b) noexcept { return a + b; });
}
template<typename T>
[[nodiscard]] __device__ inline auto lc_warp_active_product_impl(T x) noexcept {
    return lc_warp_active_reduce_impl(x, [](T a, T b) noexcept { return a * b; });
}

#define LC_WARP_ACTIVE_REDUCE_SCALAR(op, T)                                       \
    [[nodiscard]] __device__ inline auto lc_warp_active_##op(lc_##T x) noexcept { \
        return lc_warp_active_##op##_impl<lc_##T>(x);                             \
    }

#if __CUDA_ARCH__ >= 800
[[nodiscard]] __device__ inline auto lc_warp_active_min(lc_uint x) noexcept {
    return __reduce_min_sync(LC_WARP_ACTIVE_MASK, x);
}
[[nodiscard]] __device__ inline auto lc_warp_active_max(lc_uint x) noexcept {
    return __reduce_max_sync(LC_WARP_ACTIVE_MASK, x);
}
[[nodiscard]] __device__ inline auto lc_warp_active_sum(lc_uint x) noexcept {
    return __reduce_add_sync(LC_WARP_ACTIVE_MASK, x);
}
[[nodiscard]] __device__ inline auto lc_warp_active_min(lc_int x) noexcept {
    return __reduce_min_sync(LC_WARP_ACTIVE_MASK, x);
}
[[nodiscard]] __device__ inline auto lc_warp_active_max(lc_int x) noexcept {
    return __reduce_max_sync(LC_WARP_ACTIVE_MASK, x);
}
[[nodiscard]] __device__ inline auto lc_warp_active_sum(lc_int x) noexcept {
    return __reduce_add_sync(LC_WARP_ACTIVE_MASK, x);
}
[[nodiscard]] __device__ inline auto lc_warp_active_min(lc_ushort x) noexcept {
    return static_cast<lc_ushort>(__reduce_min_sync(LC_WARP_ACTIVE_MASK, static_cast<lc_uint>(x)));
}
[[nodiscard]] __device__ inline auto lc_warp_active_max(lc_ushort x) noexcept {
    return static_cast<lc_ushort>(__reduce_max_sync(LC_WARP_ACTIVE_MASK, static_cast<lc_uint>(x)));
}
[[nodiscard]] __device__ inline auto lc_warp_active_sum(lc_ushort x) noexcept {
    return static_cast<lc_ushort>(__reduce_add_sync(LC_WARP_ACTIVE_MASK, static_cast<lc_uint>(x)));
}
[[nodiscard]] __device__ inline auto lc_warp_active_min(lc_short x) noexcept {
    return static_cast<lc_short>(__reduce_min_sync(LC_WARP_ACTIVE_MASK, static_cast<lc_int>(x)));
}
[[nodiscard]] __device__ inline auto lc_warp_active_max(lc_short x) noexcept {
    return static_cast<lc_short>(__reduce_max_sync(LC_WARP_ACTIVE_MASK, static_cast<lc_int>(x)));
}
[[nodiscard]] __device__ inline auto lc_warp_active_sum(lc_short x) noexcept {
    return static_cast<lc_short>(__reduce_add_sync(LC_WARP_ACTIVE_MASK, static_cast<lc_int>(x)));
}
#else
LC_WARP_ACTIVE_REDUCE_SCALAR(min, uint)
LC_WARP_ACTIVE_REDUCE_SCALAR(max, uint)
LC_WARP_ACTIVE_REDUCE_SCALAR(sum, uint)
LC_WARP_ACTIVE_REDUCE_SCALAR(min, int)
LC_WARP_ACTIVE_REDUCE_SCALAR(max, int)
LC_WARP_ACTIVE_REDUCE_SCALAR(sum, int)
LC_WARP_ACTIVE_REDUCE_SCALAR(min, ushort)
LC_WARP_ACTIVE_REDUCE_SCALAR(max, ushort)
LC_WARP_ACTIVE_REDUCE_SCALAR(sum, ushort)
LC_WARP_ACTIVE_REDUCE_SCALAR(min, short)
LC_WARP_ACTIVE_REDUCE_SCALAR(max, short)
LC_WARP_ACTIVE_REDUCE_SCALAR(sum, short)
#endif

LC_WARP_ACTIVE_REDUCE_SCALAR(product, uint)
LC_WARP_ACTIVE_REDUCE_SCALAR(product, int)
LC_WARP_ACTIVE_REDUCE_SCALAR(product, ushort)
LC_WARP_ACTIVE_REDUCE_SCALAR(product, short)
LC_WARP_ACTIVE_REDUCE_SCALAR(min, ulong)
LC_WARP_ACTIVE_REDUCE_SCALAR(max, ulong)
LC_WARP_ACTIVE_REDUCE_SCALAR(sum, ulong)
LC_WARP_ACTIVE_REDUCE_SCALAR(product, ulong)
LC_WARP_ACTIVE_REDUCE_SCALAR(min, long)
LC_WARP_ACTIVE_REDUCE_SCALAR(max, long)
LC_WARP_ACTIVE_REDUCE_SCALAR(sum, long)
LC_WARP_ACTIVE_REDUCE_SCALAR(product, long)
LC_WARP_ACTIVE_REDUCE_SCALAR(min, float)
LC_WARP_ACTIVE_REDUCE_SCALAR(max, float)
LC_WARP_ACTIVE_REDUCE_SCALAR(sum, float)
LC_WARP_ACTIVE_REDUCE_SCALAR(product, float)
// TODO: half and double
// LC_WARP_ACTIVE_REDUCE_SCALAR(min, half)
// LC_WARP_ACTIVE_REDUCE_SCALAR(max, half)
// LC_WARP_ACTIVE_REDUCE_SCALAR(sum, half)
// LC_WARP_ACTIVE_REDUCE_SCALAR(product, half)
// LC_WARP_ACTIVE_REDUCE_SCALAR(min, double)
// LC_WARP_ACTIVE_REDUCE_SCALAR(max, double)
// LC_WARP_ACTIVE_REDUCE_SCALAR(sum, double)
// LC_WARP_ACTIVE_REDUCE_SCALAR(product, double)

#undef LC_WARP_ACTIVE_REDUCE_SCALAR

#define LC_WARP_ACTIVE_REDUCE_VECTOR2(op, T)                                         \
    [[nodiscard]] __device__ inline auto lc_warp_active_##op(lc_##T##2 v) noexcept { \
        return lc_make_##T##2(lc_warp_active_##op(v.x),                              \
                              lc_warp_active_##op(v.y));                             \
    }

#define LC_WARP_ACTIVE_REDUCE_VECTOR3(op, T)                                         \
    [[nodiscard]] __device__ inline auto lc_warp_active_##op(lc_##T##3 v) noexcept { \
        return lc_make_##T##3(lc_warp_active_##op(v.x),                              \
                              lc_warp_active_##op(v.y),                              \
                              lc_warp_active_##op(v.z));                             \
    }

#define LC_WARP_ACTIVE_REDUCE_VECTOR4(op, T)                                         \
    [[nodiscard]] __device__ inline auto lc_warp_active_##op(lc_##T##4 v) noexcept { \
        return lc_make_##T##4(lc_warp_active_##op(v.x),                              \
                              lc_warp_active_##op(v.y),                              \
                              lc_warp_active_##op(v.z),                              \
                              lc_warp_active_##op(v.w));                             \
    }

#define LC_WARP_ACTIVE_REDUCE(T)              \
    LC_WARP_ACTIVE_REDUCE_VECTOR2(min, T)     \
    LC_WARP_ACTIVE_REDUCE_VECTOR3(min, T)     \
    LC_WARP_ACTIVE_REDUCE_VECTOR4(min, T)     \
    LC_WARP_ACTIVE_REDUCE_VECTOR2(max, T)     \
    LC_WARP_ACTIVE_REDUCE_VECTOR3(max, T)     \
    LC_WARP_ACTIVE_REDUCE_VECTOR4(max, T)     \
    LC_WARP_ACTIVE_REDUCE_VECTOR2(sum, T)     \
    LC_WARP_ACTIVE_REDUCE_VECTOR3(sum, T)     \
    LC_WARP_ACTIVE_REDUCE_VECTOR4(sum, T)     \
    LC_WARP_ACTIVE_REDUCE_VECTOR2(product, T) \
    LC_WARP_ACTIVE_REDUCE_VECTOR3(product, T) \
    LC_WARP_ACTIVE_REDUCE_VECTOR4(product, T)

LC_WARP_ACTIVE_REDUCE(uint)
LC_WARP_ACTIVE_REDUCE(int)
LC_WARP_ACTIVE_REDUCE(ushort)
LC_WARP_ACTIVE_REDUCE(short)
LC_WARP_ACTIVE_REDUCE(ulong)
LC_WARP_ACTIVE_REDUCE(long)
LC_WARP_ACTIVE_REDUCE(float)
//LC_WARP_ACTIVE_REDUCE(half)// TODO
//LC_WARP_ACTIVE_REDUCE(double)// TODO

#undef LC_WARP_ACTIVE_REDUCE_VECTOR2
#undef LC_WARP_ACTIVE_REDUCE_VECTOR3
#undef LC_WARP_ACTIVE_REDUCE_VECTOR4
#undef LC_WARP_ACTIVE_REDUCE

[[nodiscard]] __device__ inline auto lc_warp_prev_active_lane() noexcept {
    auto mask = 0u;
    asm("mov.u32 %0, %lanemask_lt;"
        : "=r"(mask));
    return (lc_warp_size() - 1u) - __clz(LC_WARP_ACTIVE_MASK & mask);
}

template<typename T, typename F>
[[nodiscard]] __device__ inline auto lc_warp_prefix_reduce_impl(T x, T unit, F f) noexcept {
    auto mask = LC_WARP_ACTIVE_MASK;
    auto lane = lc_warp_lane_id();
    x = __shfl_sync(mask, x, lc_warp_prev_active_lane());
    x = (lane == lc_warp_first_active_lane()) ? unit : x;
    if (auto y = __shfl_up_sync(mask, x, 0x01u); lane >= 0x01u && (mask & (1u << (lane - 0x01u)))) { x = f(x, y); }
    if (auto y = __shfl_up_sync(mask, x, 0x02u); lane >= 0x02u && (mask & (1u << (lane - 0x02u)))) { x = f(x, y); }
    if (auto y = __shfl_up_sync(mask, x, 0x04u); lane >= 0x04u && (mask & (1u << (lane - 0x04u)))) { x = f(x, y); }
    if (auto y = __shfl_up_sync(mask, x, 0x08u); lane >= 0x08u && (mask & (1u << (lane - 0x08u)))) { x = f(x, y); }
    if (auto y = __shfl_up_sync(mask, x, 0x10u); lane >= 0x10u && (mask & (1u << (lane - 0x10u)))) { x = f(x, y); }
    return x;
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_warp_prefix_sum_impl(T x) noexcept {
    return lc_warp_prefix_reduce_impl(x, static_cast<T>(0), [](T a, T b) noexcept { return a + b; });
}

template<typename T>
[[nodiscard]] __device__ inline auto lc_warp_prefix_product_impl(T x) noexcept {
    return lc_warp_prefix_reduce_impl(x, static_cast<T>(1), [](T a, T b) noexcept { return a * b; });
}

#define LC_WARP_PREFIX_REDUCE_SCALAR(op, T)                                       \
    [[nodiscard]] __device__ inline auto lc_warp_prefix_##op(lc_##T x) noexcept { \
        return lc_warp_prefix_##op##_impl<lc_##T>(x);                             \
    }

#define LC_WARP_PREFIX_REDUCE_VECTOR2(op, T)                                         \
    [[nodiscard]] __device__ inline auto lc_warp_prefix_##op(lc_##T##2 v) noexcept { \
        return lc_make_##T##2(lc_warp_prefix_##op(v.x),                              \
                              lc_warp_prefix_##op(v.y));                             \
    }

#define LC_WARP_PREFIX_REDUCE_VECTOR3(op, T)                                         \
    [[nodiscard]] __device__ inline auto lc_warp_prefix_##op(lc_##T##3 v) noexcept { \
        return lc_make_##T##3(lc_warp_prefix_##op(v.x),                              \
                              lc_warp_prefix_##op(v.y),                              \
                              lc_warp_prefix_##op(v.z));                             \
    }

#define LC_WARP_PREFIX_REDUCE_VECTOR4(op, T)                                         \
    [[nodiscard]] __device__ inline auto lc_warp_prefix_##op(lc_##T##4 v) noexcept { \
        return lc_make_##T##4(lc_warp_prefix_##op(v.x),                              \
                              lc_warp_prefix_##op(v.y),                              \
                              lc_warp_prefix_##op(v.z),                              \
                              lc_warp_prefix_##op(v.w));                             \
    }

#define LC_WARP_PREFIX_REDUCE(T)              \
    LC_WARP_PREFIX_REDUCE_SCALAR(sum, T)      \
    LC_WARP_PREFIX_REDUCE_SCALAR(product, T)  \
    LC_WARP_PREFIX_REDUCE_VECTOR2(sum, T)     \
    LC_WARP_PREFIX_REDUCE_VECTOR2(product, T) \
    LC_WARP_PREFIX_REDUCE_VECTOR3(sum, T)     \
    LC_WARP_PREFIX_REDUCE_VECTOR3(product, T) \
    LC_WARP_PREFIX_REDUCE_VECTOR4(sum, T)     \
    LC_WARP_PREFIX_REDUCE_VECTOR4(product, T)

LC_WARP_PREFIX_REDUCE(uint)
LC_WARP_PREFIX_REDUCE(int)
LC_WARP_PREFIX_REDUCE(ushort)
LC_WARP_PREFIX_REDUCE(short)
LC_WARP_PREFIX_REDUCE(ulong)
LC_WARP_PREFIX_REDUCE(long)
LC_WARP_PREFIX_REDUCE(float)
//LC_WARP_PREFIX_REDUCE(half)// TODO
//LC_WARP_PREFIX_REDUCE(double)// TODO

#undef LC_WARP_PREFIX_REDUCE_SCALAR
#undef LC_WARP_PREFIX_REDUCE_VECTOR2
#undef LC_WARP_PREFIX_REDUCE_VECTOR3
#undef LC_WARP_PREFIX_REDUCE_VECTOR4
#undef LC_WARP_PREFIX_REDUCE

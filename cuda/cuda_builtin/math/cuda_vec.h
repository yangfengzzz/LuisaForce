//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "math/cuda_initializer_array.h"
#include "math/cuda_math_utils.h"

namespace wp {

template<unsigned Length, typename Type>
struct vec_t {
    Type c[Length] = {};

    inline vec_t() = default;

    inline CUDA_CALLABLE vec_t(Type s) {
        for (unsigned i = 0; i < Length; ++i) {
            c[i] = s;
        }
    }

    template<unsigned OtherLength, typename OtherType>
    inline explicit CUDA_CALLABLE vec_t(const vec_t<OtherLength, OtherType> &other) {
        for (unsigned i = 0; i < Length; ++i) {
            c[i] = other[i];
        }
    }

    inline CUDA_CALLABLE constexpr vec_t(Type x, Type y) {
        assert(Length == 2);
        c[0] = x;
        c[1] = y;
    }

    inline CUDA_CALLABLE constexpr vec_t(Type x, Type y, Type z) {
        assert(Length == 3);
        c[0] = x;
        c[1] = y;
        c[2] = z;
    }

    inline CUDA_CALLABLE constexpr vec_t(Type x, Type y, Type z, Type w) {
        assert(Length == 4);
        c[0] = x;
        c[1] = y;
        c[2] = z;
        c[3] = w;
    }

    inline CUDA_CALLABLE vec_t(const initializer_array<Length, Type> &l) {
        for (unsigned i = 0; i < Length; ++i) {
            c[i] = l[i];
        }
    }

    // special screw vector constructor for spatial_vectors:
    template<typename OtherType>
    inline CUDA_CALLABLE vec_t(vec_t<3, OtherType> w, vec_t<3, OtherType> v) {
        c[0] = w[0];
        c[1] = w[1];
        c[2] = w[2];
        c[3] = v[0];
        c[4] = v[1];
        c[5] = v[2];
    }

    template<typename OtherType>
    inline CUDA_CALLABLE vec_t(OtherType x, vec_t<2, OtherType> yz) {
        c[0] = x;
        c[1] = yz[0];
        c[2] = yz[1];
    }

    template<typename OtherType>
    inline CUDA_CALLABLE vec_t(vec_t<2, OtherType> xy, OtherType z) {
        c[0] = xy[0];
        c[1] = xy[1];
        c[2] = z;
    }

    template<typename OtherType>
    inline CUDA_CALLABLE vec_t(OtherType x, OtherType y, vec_t<2, OtherType> zw) {
        c[0] = x;
        c[1] = y;
        c[2] = zw[0];
        c[3] = zw[1];
    }

    template<typename OtherType>
    inline CUDA_CALLABLE vec_t(OtherType x, vec_t<2, OtherType> yz, OtherType w) {
        c[0] = x;
        c[1] = yz[0];
        c[2] = yz[1];
        c[3] = w;
    }

    template<typename OtherType>
    inline CUDA_CALLABLE vec_t(vec_t<2, OtherType> xy, OtherType z, OtherType w) {
        c[0] = xy[0];
        c[1] = xy[1];
        c[2] = z;
        c[3] = w;
    }

    template<typename OtherType>
    inline CUDA_CALLABLE vec_t(vec_t<2, OtherType> xy, vec_t<2, OtherType> zw) {
        c[0] = xy[0];
        c[1] = xy[1];
        c[2] = zw[0];
        c[3] = zw[1];
    }

    template<typename OtherType>
    inline CUDA_CALLABLE vec_t(OtherType x, vec_t<3, OtherType> yzw) {
        c[0] = x;
        c[1] = yzw[0];
        c[2] = yzw[1];
        c[3] = yzw[2];
    }

    template<typename OtherType>
    inline CUDA_CALLABLE vec_t(vec_t<3, OtherType> xyz, OtherType w) {
        c[0] = xyz[0];
        c[1] = xyz[1];
        c[2] = xyz[2];
        c[3] = w;
    }

    inline CUDA_CALLABLE Type operator[](int index) const {
        assert(index < Length);
        return c[index];
    }

    inline CUDA_CALLABLE Type &operator[](int index) {
        assert(index < Length);
        return c[index];
    }

    inline CUDA_CALLABLE constexpr Type x() const {
        return c[0];
    }

    inline CUDA_CALLABLE constexpr Type x() {
        return c[0];
    }

    inline CUDA_CALLABLE constexpr Type y() const {
        assert(1 < Length);
        return c[1];
    }

    inline CUDA_CALLABLE constexpr Type z() const {
        assert(2 < Length);
        return c[2];
    }

    inline CUDA_CALLABLE constexpr Type w() const {
        assert(3 < Length);
        return c[3];
    }
};

using vec2b = vec_t<2, int8>;
using vec3b = vec_t<3, int8>;
using vec4b = vec_t<4, int8>;
using vec2ub = vec_t<2, uint8>;
using vec3ub = vec_t<3, uint8>;
using vec4ub = vec_t<4, uint8>;

using vec2s = vec_t<2, int16>;
using vec3s = vec_t<3, int16>;
using vec4s = vec_t<4, int16>;
using vec2us = vec_t<2, uint16>;
using vec3us = vec_t<3, uint16>;
using vec4us = vec_t<4, uint16>;

using vec2i = vec_t<2, int32>;
using vec3i = vec_t<3, int32>;
using vec4i = vec_t<4, int32>;
using vec2ui = vec_t<2, uint32>;
using vec3ui = vec_t<3, uint32>;
using vec4ui = vec_t<4, uint32>;

using vec2l = vec_t<2, int64>;
using vec3l = vec_t<3, int64>;
using vec4l = vec_t<4, int64>;
using vec2ul = vec_t<2, uint64>;
using vec3ul = vec_t<3, uint64>;
using vec4ul = vec_t<4, uint64>;

using vec2h = vec_t<2, half>;
using vec3h = vec_t<3, half>;
using vec4h = vec_t<4, half>;

using vec2 = vec_t<2, float>;
using vec3 = vec_t<3, float>;
using vec4 = vec_t<4, float>;

using vec2f = vec_t<2, float>;
using vec3f = vec_t<3, float>;
using vec4f = vec_t<4, float>;

using vec2d = vec_t<2, double>;
using vec3d = vec_t<3, double>;
using vec4d = vec_t<4, double>;

using wp_short2 = vec_t<2, wp_short>;
using wp_short3 = vec_t<3, wp_short>;
using wp_short4 = vec_t<4, wp_short>;

using wp_ushort2 = vec_t<2, wp_ushort>;
using wp_ushort3 = vec_t<3, wp_ushort>;
using wp_ushort4 = vec_t<4, wp_ushort>;

using wp_int2 = vec_t<2, wp_int>;
using wp_int3 = vec_t<3, wp_int>;
using wp_int4 = vec_t<4, wp_int>;

using wp_uint2 = vec_t<2, wp_uint>;
using wp_uint3 = vec_t<3, wp_uint>;
using wp_uint4 = vec_t<4, wp_uint>;

using wp_float2 = vec_t<2, wp_float>;
using wp_float3 = vec_t<3, wp_float>;
using wp_float4 = vec_t<4, wp_float>;

using wp_bool2 = vec_t<2, wp_bool>;
using wp_bool3 = vec_t<3, wp_bool>;
using wp_bool4 = vec_t<4, wp_bool>;

using wp_long2 = vec_t<2, wp_long>;
using wp_long3 = vec_t<3, wp_long>;
using wp_long4 = vec_t<4, wp_long>;

using wp_ulong2 = vec_t<2, wp_ulong>;
using wp_ulong3 = vec_t<3, wp_ulong>;
using wp_ulong4 = vec_t<4, wp_ulong>;

//--------------
// vec<Length, Type> methods

// Should these accept const references as arguments? It's all
// inlined so maybe it doesn't matter? Even if it does, it
// probably depends on the Length of the vector...

// negation:
template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> operator-(vec_t<Length, Type> a) {
    // NB: this constructor will initialize all ret's components to 0, which is
    // unnecessary...
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = -a[i];
    }

    // Wonder if this does a load of copying when it returns... hopefully not as it's inlined?
    return ret;
}

template<unsigned Length, typename Type>
CUDA_CALLABLE inline vec_t<Length, Type> pos(const vec_t<Length, Type> &x) {
    return x;
}

template<unsigned Length, typename Type>
CUDA_CALLABLE inline vec_t<Length, Type> neg(const vec_t<Length, Type> &x) {
    return -x;
}

template<typename Type>
CUDA_CALLABLE inline vec_t<3, Type> neg(const vec_t<3, Type> &x) {
    return vec_t<3, Type>(-x.c[0], -x.c[1], -x.c[2]);
}

template<typename Type>
CUDA_CALLABLE inline vec_t<2, Type> neg(const vec_t<2, Type> &x) {
    return vec_t<2, Type>(-x.c[0], -x.c[1]);
}

// equality:
template<unsigned Length, typename Type>
inline CUDA_CALLABLE bool operator==(const vec_t<Length, Type> &a, const vec_t<Length, Type> &b) {
    for (unsigned i = 0; i < Length; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

// scalar multiplication:
template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> mul(vec_t<Length, Type> a, Type s) {
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] * s;
    }
    return ret;
}

template<typename Type>
inline CUDA_CALLABLE vec_t<3, Type> mul(vec_t<3, Type> a, Type s) {
    return vec_t<3, Type>(a.c[0] * s, a.c[1] * s, a.c[2] * s);
}

template<typename Type>
inline CUDA_CALLABLE vec_t<2, Type> mul(vec_t<2, Type> a, Type s) {
    return vec_t<2, Type>(a.c[0] * s, a.c[1] * s);
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> mul(Type s, vec_t<Length, Type> a) {
    return mul(a, s);
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> operator*(Type s, vec_t<Length, Type> a) {
    return mul(a, s);
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> operator*(vec_t<Length, Type> a, Type s) {
    return mul(a, s);
}

// component wise multiplication:
template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> cw_mul(vec_t<Length, Type> a, vec_t<Length, Type> b) {
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] * b[i];
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> operator*(vec_t<Length, Type> a, vec_t<Length, Type> b) {
    return cw_mul(a, b);
}

// division
template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> div(vec_t<Length, Type> a, Type s) {
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] / s;
    }
    return ret;
}

template<typename Type>
inline CUDA_CALLABLE vec_t<3, Type> div(vec_t<3, Type> a, Type s) {
    return vec_t<3, Type>(a.c[0] / s, a.c[1] / s, a.c[2] / s);
}

template<typename Type>
inline CUDA_CALLABLE vec_t<2, Type> div(vec_t<2, Type> a, Type s) {
    return vec_t<2, Type>(a.c[0] / s, a.c[1] / s);
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> operator/(vec_t<Length, Type> a, Type s) {
    return div(a, s);
}

// component wise division
template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> cw_div(vec_t<Length, Type> a, vec_t<Length, Type> b) {
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] / b[i];
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> operator/(vec_t<Length, Type> a, vec_t<Length, Type> b) {
    return cw_div(a, b);
}

// addition
template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> add(vec_t<Length, Type> a, vec_t<Length, Type> b) {
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] + b[i];
    }
    return ret;
}

template<typename Type>
inline CUDA_CALLABLE vec_t<2, Type> add(vec_t<2, Type> a, vec_t<2, Type> b) {
    return vec_t<2, Type>(a.c[0] + b.c[0], a.c[1] + b.c[1]);
}

template<typename Type>
inline CUDA_CALLABLE vec_t<3, Type> add(vec_t<3, Type> a, vec_t<3, Type> b) {
    return vec_t<3, Type>(a.c[0] + b.c[0], a.c[1] + b.c[1], a.c[2] + b.c[2]);
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> operator+(vec_t<Length, Type> a, Type s) {
    return add(a, vec_t<Length, Type>(s));
}

// subtraction
template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> sub(vec_t<Length, Type> a, vec_t<Length, Type> b) {
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = Type(a[i] - b[i]);
    }
    return ret;
}

template<typename Type>
inline CUDA_CALLABLE vec_t<2, Type> sub(vec_t<2, Type> a, vec_t<2, Type> b) {
    return vec_t<2, Type>(a.c[0] - b.c[0], a.c[1] - b.c[1]);
}

template<typename Type>
inline CUDA_CALLABLE vec_t<3, Type> sub(vec_t<3, Type> a, vec_t<3, Type> b) {
    return vec_t<3, Type>(a.c[0] - b.c[0], a.c[1] - b.c[1], a.c[2] - b.c[2]);
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> operator-(vec_t<Length, Type> a, Type s) {
    return sub(a, vec_t<Length, Type>(s));
}

// cmp
template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, bool> operator<(Type s, vec_t<Length, Type> a) {
    vec_t<Length, bool> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = s < a[i];
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, bool> operator<(vec_t<Length, Type> a, Type s) {
    vec_t<Length, bool> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] < s;
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, bool> operator<(vec_t<Length, Type> a, vec_t<Length, Type> b) {
    vec_t<Length, bool> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] < b[i];
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, bool> operator<=(Type s, vec_t<Length, Type> a) {
    vec_t<Length, bool> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = s <= a[i];
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, bool> operator<=(vec_t<Length, Type> a, Type s) {
    vec_t<Length, bool> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] <= s;
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, bool> operator<=(vec_t<Length, Type> a, vec_t<Length, Type> b) {
    vec_t<Length, bool> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] <= b[i];
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, bool> operator>(Type s, vec_t<Length, Type> a) {
    vec_t<Length, bool> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = s > a[i];
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, bool> operator>(vec_t<Length, Type> a, Type s) {
    vec_t<Length, bool> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] > s;
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, bool> operator>(vec_t<Length, Type> a, vec_t<Length, Type> b) {
    vec_t<Length, bool> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] > b[i];
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, bool> operator>=(Type s, vec_t<Length, Type> a) {
    vec_t<Length, bool> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = s >= a[i];
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, bool> operator>=(vec_t<Length, Type> a, Type s) {
    vec_t<Length, bool> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] >= s;
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, bool> operator>=(vec_t<Length, Type> a, vec_t<Length, Type> b) {
    vec_t<Length, bool> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] >= b[i];
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, bool> operator==(Type s, vec_t<Length, Type> a) {
    vec_t<Length, bool> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = s == a[i];
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, bool> operator==(vec_t<Length, Type> a, Type s) {
    vec_t<Length, bool> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] == s;
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, bool> operator==(vec_t<Length, Type> a, vec_t<Length, Type> b) {
    vec_t<Length, bool> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] == b[i];
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, bool> operator!=(Type s, vec_t<Length, Type> a) {
    vec_t<Length, bool> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = s != a[i];
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, bool> operator!=(vec_t<Length, Type> a, Type s) {
    vec_t<Length, bool> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] != s;
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, bool> operator!=(vec_t<Length, Type> a, vec_t<Length, Type> b) {
    vec_t<Length, bool> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] != b[i];
    }
    return ret;
}

// dot product:
template<unsigned Length, typename Type>
inline CUDA_CALLABLE Type dot(vec_t<Length, Type> a, vec_t<Length, Type> b) {
    Type ret(0);
    for (unsigned i = 0; i < Length; ++i) {
        ret += a[i] * b[i];
    }
    return ret;
}

template<typename Type>
inline CUDA_CALLABLE Type dot(vec_t<2, Type> a, vec_t<2, Type> b) {
    return a.c[0] * b.c[0] + a.c[1] * b.c[1];
}

template<typename Type>
inline CUDA_CALLABLE Type dot(vec_t<3, Type> a, vec_t<3, Type> b) {
    return a.c[0] * b.c[0] + a.c[1] * b.c[1] + a.c[2] * b.c[2];
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE Type tensordot(vec_t<Length, Type> a, vec_t<Length, Type> b) {
    // corresponds to `np.tensordot()` with all axes being contracted
    return dot(a, b);
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE Type index(const vec_t<Length, Type> &a, int idx) {
#ifndef NDEBUG
    if (idx < 0 || idx >= Length) {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    return a[idx];
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void indexset(vec_t<Length, Type> &v, int idx, Type value) {
#ifndef NDEBUG
    if (idx < 0 || idx >= Length) {
        printf("vec store %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    v[idx] = value;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE Type length(vec_t<Length, Type> a) {
    return sqrt(dot(a, a));
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE Type length_sq(vec_t<Length, Type> a) {
    return dot(a, a);
}

template<typename Type>
inline CUDA_CALLABLE Type length(vec_t<2, Type> a) {
    return sqrt(a.c[0] * a.c[0] + a.c[1] * a.c[1]);
}

template<typename Type>
inline CUDA_CALLABLE Type length(vec_t<3, Type> a) {
    return sqrt(a.c[0] * a.c[0] + a.c[1] * a.c[1] + a.c[2] * a.c[2]);
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> normalize(vec_t<Length, Type> a) {
    Type l = length(a);
    if (l > Type(kEps))
        return div(a, l);
    else
        return vec_t<Length, Type>();
}

template<typename Type>
inline CUDA_CALLABLE vec_t<2, Type> normalize(vec_t<2, Type> a) {
    Type l = sqrt(a.c[0] * a.c[0] + a.c[1] * a.c[1]);
    if (l > Type(kEps))
        return vec_t<2, Type>(a.c[0] / l, a.c[1] / l);
    else
        return vec_t<2, Type>();
}

template<typename Type>
inline CUDA_CALLABLE vec_t<3, Type> normalize(vec_t<3, Type> a) {
    Type l = sqrt(a.c[0] * a.c[0] + a.c[1] * a.c[1] + a.c[2] * a.c[2]);
    if (l > Type(kEps))
        return vec_t<3, Type>(a.c[0] / l, a.c[1] / l, a.c[2] / l);
    else
        return vec_t<3, Type>();
}

template<typename Type>
inline CUDA_CALLABLE vec_t<3, Type> cross(vec_t<3, Type> a, vec_t<3, Type> b) {
    return {
        Type(a[1] * b[2] - a[2] * b[1]),
        Type(a[2] * b[0] - a[0] * b[2]),
        Type(a[0] * b[1] - a[1] * b[0])};
}

template<unsigned Length, typename Type>
inline bool CUDA_CALLABLE isfinite(vec_t<Length, Type> x) {
    for (unsigned i = 0; i < Length; ++i) {
        if (!isfinite(x[i])) {
            return false;
        }
    }
    return true;
}

// These two functions seem to compile very slowly
template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> min(vec_t<Length, Type> a, vec_t<Length, Type> b) {
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] < b[i] ? a[i] : b[i];
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> max(vec_t<Length, Type> a, vec_t<Length, Type> b) {
    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] > b[i] ? a[i] : b[i];
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE Type min(vec_t<Length, Type> v) {
    Type ret = v[0];
    for (unsigned i = 1; i < Length; ++i) {
        if (v[i] < ret)
            ret = v[i];
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE Type max(vec_t<Length, Type> v) {
    Type ret = v[0];
    for (unsigned i = 1; i < Length; ++i) {
        if (v[i] > ret)
            ret = v[i];
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE unsigned argmin(vec_t<Length, Type> v) {
    unsigned ret = 0;
    for (unsigned i = 1; i < Length; ++i) {
        if (v[i] < v[ret])
            ret = i;
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE unsigned argmax(vec_t<Length, Type> v) {
    unsigned ret = 0;
    for (unsigned i = 1; i < Length; ++i) {
        if (v[i] > v[ret])
            ret = i;
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void expect_near(const vec_t<Length, Type> &actual, const vec_t<Length, Type> &expected, const Type &tolerance) {
    const Type diff(0);
    for (size_t i = 0; i < Length; ++i) {
        diff = max(diff, abs(actual[i] - expected[i]));
    }
    if (diff > tolerance) {
        printf("Error, expect_near() failed with tolerance ");
        print(tolerance);
        printf("\t Expected: ");
        print(expected);
        printf("\t Actual: ");
        print(actual);
    }
}

// Do I need to specialize these for different lengths?
template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> atomic_add(vec_t<Length, Type> *addr, vec_t<Length, Type> value) {

    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = atomic_add(&(addr->c[i]), value[i]);
    }

    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> atomic_min(vec_t<Length, Type> *addr, vec_t<Length, Type> value) {

    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = atomic_min(&(addr->c[i]), value[i]);
    }

    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> atomic_max(vec_t<Length, Type> *addr, vec_t<Length, Type> value) {

    vec_t<Length, Type> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = atomic_max(&(addr->c[i]), value[i]);
    }

    return ret;
}

// ok, the original implementation of this didn't take the absolute values.
// I wouldn't consider this expected behavior. It looks like it's only
// being used for bounding boxes at the moment, where this doesn't matter,
// but you often use it for ray tracing where it does. Not sure if the
// fabs() incurs a performance hit...
template<unsigned Length, typename Type>
CUDA_CALLABLE inline int longest_axis(const vec_t<Length, Type> &v) {
    Type lmax = abs(v[0]);
    int ret(0);
    for (unsigned i = 1; i < Length; ++i) {
        Type l = abs(v[i]);
        if (l > lmax) {
            ret = i;
            lmax = l;
        }
    }
    return ret;
}

template<unsigned Length, typename Type>
CUDA_CALLABLE inline vec_t<Length, Type> lerp(const vec_t<Length, Type> &a, const vec_t<Length, Type> &b, Type t) {
    return a * (Type(1) - t) + b * t;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void print(vec_t<Length, Type> v) {
    for (unsigned i = 0; i < Length; ++i) {
        printf("%g ", float(v[i]));
    }
    printf("\n");
}

inline CUDA_CALLABLE void expect_near(const vec3 &actual, const vec3 &expected, const float &tolerance) {
    const float diff = max(max(abs(actual[0] - expected[0]), abs(actual[1] - expected[1])), abs(actual[2] - expected[2]));
    if (diff > tolerance) {
        printf("Error, expect_near() failed with tolerance ");
        print(tolerance);
        printf("\t Expected: ");
        print(expected);
        printf("\t Actual: ");
        print(actual);
    }
}

CUDA_CALLABLE_DEVICE inline constexpr auto any(vec_t<2, bool> v) noexcept { return v.x() || v.y(); }
CUDA_CALLABLE_DEVICE inline constexpr auto any(vec_t<3, bool> v) noexcept { return v.x() || v.y() || v.z(); }
CUDA_CALLABLE_DEVICE inline constexpr auto any(vec_t<4, bool> v) noexcept { return v.x() || v.y() || v.z() || v.w(); }
CUDA_CALLABLE_DEVICE inline constexpr auto all(vec_t<2, bool> v) noexcept { return v.x() && v.y(); }
CUDA_CALLABLE_DEVICE inline constexpr auto all(vec_t<3, bool> v) noexcept { return v.x() && v.y() && v.z(); }
CUDA_CALLABLE_DEVICE inline constexpr auto all(vec_t<4, bool> v) noexcept { return v.x() && v.y() && v.z() && v.w(); }
CUDA_CALLABLE_DEVICE inline constexpr auto none(vec_t<2, bool> v) noexcept { return !v.x() && !v.y(); }
CUDA_CALLABLE_DEVICE inline constexpr auto none(vec_t<3, bool> v) noexcept { return !v.x() && !v.y() && !v.z(); }
CUDA_CALLABLE_DEVICE inline constexpr auto none(vec_t<4, bool> v) noexcept { return !v.x() && !v.y() && !v.z() && !v.w(); }

CUDA_CALLABLE_DEVICE inline wp_float2 abs(wp_float2 x) noexcept { return wp_float2(fabsf(x.x()), fabsf(x.y())); }
CUDA_CALLABLE_DEVICE inline wp_float3 abs(wp_float3 x) noexcept { return wp_float3(fabsf(x.x()), fabsf(x.y()), fabsf(x.z())); }
CUDA_CALLABLE_DEVICE inline wp_float4 abs(wp_float4 x) noexcept { return wp_float4(fabsf(x.x()), fabsf(x.y()), fabsf(x.z()), fabsf(x.w())); }

CUDA_CALLABLE_DEVICE inline wp_float2 acos(wp_float2 x) noexcept { return wp_float2(acosf(x.x()), acosf(x.y())); }
CUDA_CALLABLE_DEVICE inline wp_float3 acos(wp_float3 x) noexcept { return wp_float3(acosf(x.x()), acosf(x.y()), acosf(x.z())); }
CUDA_CALLABLE_DEVICE inline wp_float4 acos(wp_float4 x) noexcept { return wp_float4(acosf(x.x()), acosf(x.y()), acosf(x.z()), acosf(x.w())); }

CUDA_CALLABLE_DEVICE inline wp_float2 asin(wp_float2 x) noexcept { return wp_float2(asinf(x.x()), asinf(x.y())); }
CUDA_CALLABLE_DEVICE inline wp_float3 asin(wp_float3 x) noexcept { return wp_float3(asinf(x.x()), asinf(x.y()), asinf(x.z())); }
CUDA_CALLABLE_DEVICE inline wp_float4 asin(wp_float4 x) noexcept { return wp_float4(asinf(x.x()), asinf(x.y()), asinf(x.z()), asinf(x.w())); }

CUDA_CALLABLE_DEVICE inline wp_float2 atan(wp_float2 x) noexcept { return wp_float2(atanf(x.x()), atanf(x.y())); }
CUDA_CALLABLE_DEVICE inline wp_float3 atan(wp_float3 x) noexcept { return wp_float3(atanf(x.x()), atanf(x.y()), atanf(x.z())); }
CUDA_CALLABLE_DEVICE inline wp_float4 atan(wp_float4 x) noexcept { return wp_float4(atanf(x.x()), atanf(x.y()), atanf(x.z()), atanf(x.w())); }

CUDA_CALLABLE_DEVICE inline float acosh(wp_float x) noexcept { return acoshf(x); }
CUDA_CALLABLE_DEVICE inline wp_float2 acosh(wp_float2 x) noexcept { return wp_float2(acoshf(x.x()), acoshf(x.y())); }
CUDA_CALLABLE_DEVICE inline wp_float3 acosh(wp_float3 x) noexcept { return wp_float3(acoshf(x.x()), acoshf(x.y()), acoshf(x.z())); }
CUDA_CALLABLE_DEVICE inline wp_float4 acosh(wp_float4 x) noexcept { return wp_float4(acoshf(x.x()), acoshf(x.y()), acoshf(x.z()), acoshf(x.w())); }

CUDA_CALLABLE_DEVICE inline float asinh(wp_float x) noexcept { return asinhf(x); }
CUDA_CALLABLE_DEVICE inline wp_float2 asinh(wp_float2 x) noexcept { return wp_float2(asinhf(x.x()), asinhf(x.y())); }
CUDA_CALLABLE_DEVICE inline wp_float3 asinh(wp_float3 x) noexcept { return wp_float3(asinhf(x.x()), asinhf(x.y()), asinhf(x.z())); }
CUDA_CALLABLE_DEVICE inline wp_float4 asinh(wp_float4 x) noexcept { return wp_float4(asinhf(x.x()), asinhf(x.y()), asinhf(x.z()), asinhf(x.w())); }

CUDA_CALLABLE_DEVICE inline auto atanh(wp_float x) noexcept { return atanhf(x); }
CUDA_CALLABLE_DEVICE inline auto atanh(wp_float2 x) noexcept { return wp_float2(atanhf(x.x()), atanhf(x.y())); }
CUDA_CALLABLE_DEVICE inline auto atanh(wp_float3 x) noexcept { return wp_float3(atanhf(x.x()), atanhf(x.y()), atanhf(x.z())); }
CUDA_CALLABLE_DEVICE inline auto atanh(wp_float4 x) noexcept { return wp_float4(atanhf(x.x()), atanhf(x.y()), atanhf(x.z()), atanhf(x.w())); }

CUDA_CALLABLE_DEVICE inline auto atan2(wp_float2 y, wp_float2 x) noexcept { return wp_float2(atan2f(y.x(), x.x()), atan2f(y.y(), x.y())); }
CUDA_CALLABLE_DEVICE inline auto atan2(wp_float3 y, wp_float3 x) noexcept { return wp_float3(atan2f(y.x(), x.x()), atan2f(y.y(), x.y()), atan2f(y.z(), x.z())); }
CUDA_CALLABLE_DEVICE inline auto atan2(wp_float4 y, wp_float4 x) noexcept { return wp_float4(atan2f(y.x(), x.x()), atan2f(y.y(), x.y()), atan2f(y.z(), x.z()), atan2f(y.w(), x.w())); }

CUDA_CALLABLE_DEVICE inline auto cos(wp_float2 x) noexcept { return wp_float2(cosf(x.x()), cosf(x.y())); }
CUDA_CALLABLE_DEVICE inline auto cos(wp_float3 x) noexcept { return wp_float3(cosf(x.x()), cosf(x.y()), cosf(x.z())); }
CUDA_CALLABLE_DEVICE inline auto cos(wp_float4 x) noexcept { return wp_float4(cosf(x.x()), cosf(x.y()), cosf(x.z()), cosf(x.w())); }

CUDA_CALLABLE_DEVICE inline auto cosh(wp_float2 x) noexcept { return wp_float2(coshf(x.x()), coshf(x.y())); }
CUDA_CALLABLE_DEVICE inline auto cosh(wp_float3 x) noexcept { return wp_float3(coshf(x.x()), coshf(x.y()), coshf(x.z())); }
CUDA_CALLABLE_DEVICE inline auto cosh(wp_float4 x) noexcept { return wp_float4(coshf(x.x()), coshf(x.y()), coshf(x.z()), coshf(x.w())); }

CUDA_CALLABLE_DEVICE inline auto sin(wp_float2 x) noexcept { return wp_float2(sinf(x.x()), sinf(x.y())); }
CUDA_CALLABLE_DEVICE inline auto sin(wp_float3 x) noexcept { return wp_float3(sinf(x.x()), sinf(x.y()), sinf(x.z())); }
CUDA_CALLABLE_DEVICE inline auto sin(wp_float4 x) noexcept { return wp_float4(sinf(x.x()), sinf(x.y()), sinf(x.z()), sinf(x.w())); }

CUDA_CALLABLE_DEVICE inline auto sinh(wp_float2 x) noexcept { return wp_float2(sinhf(x.x()), sinhf(x.y())); }
CUDA_CALLABLE_DEVICE inline auto sinh(wp_float3 x) noexcept { return wp_float3(sinhf(x.x()), sinhf(x.y()), sinhf(x.z())); }
CUDA_CALLABLE_DEVICE inline auto sinh(wp_float4 x) noexcept { return wp_float4(sinhf(x.x()), sinhf(x.y()), sinhf(x.z()), sinhf(x.w())); }

CUDA_CALLABLE_DEVICE inline auto tan(wp_float2 x) noexcept { return wp_float2(tanf(x.x()), tanf(x.y())); }
CUDA_CALLABLE_DEVICE inline auto tan(wp_float3 x) noexcept { return wp_float3(tanf(x.x()), tanf(x.y()), tanf(x.z())); }
CUDA_CALLABLE_DEVICE inline auto tan(wp_float4 x) noexcept { return wp_float4(tanf(x.x()), tanf(x.y()), tanf(x.z()), tanf(x.w())); }

CUDA_CALLABLE_DEVICE inline auto tanh(wp_float2 x) noexcept { return wp_float2(tanhf(x.x()), tanhf(x.y())); }
CUDA_CALLABLE_DEVICE inline auto tanh(wp_float3 x) noexcept { return wp_float3(tanhf(x.x()), tanhf(x.y()), tanhf(x.z())); }
CUDA_CALLABLE_DEVICE inline auto tanh(wp_float4 x) noexcept { return wp_float4(tanhf(x.x()), tanhf(x.y()), tanhf(x.z()), tanhf(x.w())); }

CUDA_CALLABLE_DEVICE inline auto exp(wp_float2 x) noexcept { return wp_float2(expf(x.x()), expf(x.y())); }
CUDA_CALLABLE_DEVICE inline auto exp(wp_float3 x) noexcept { return wp_float3(expf(x.x()), expf(x.y()), expf(x.z())); }
CUDA_CALLABLE_DEVICE inline auto exp(wp_float4 x) noexcept { return wp_float4(expf(x.x()), expf(x.y()), expf(x.z()), expf(x.w())); }

CUDA_CALLABLE_DEVICE inline auto exp2(wp_float x) noexcept { return exp2f(x); }
CUDA_CALLABLE_DEVICE inline auto exp2(wp_float2 x) noexcept { return wp_float2(exp2f(x.x()), exp2f(x.y())); }
CUDA_CALLABLE_DEVICE inline auto exp2(wp_float3 x) noexcept { return wp_float3(exp2f(x.x()), exp2f(x.y()), exp2f(x.z())); }
CUDA_CALLABLE_DEVICE inline auto exp2(wp_float4 x) noexcept { return wp_float4(exp2f(x.x()), exp2f(x.y()), exp2f(x.z()), exp2f(x.w())); }

CUDA_CALLABLE_DEVICE inline auto exp10(wp_float x) noexcept { return exp10f(x); }
CUDA_CALLABLE_DEVICE inline auto exp10(wp_float2 x) noexcept { return wp_float2(exp10f(x.x()), exp10f(x.y())); }
CUDA_CALLABLE_DEVICE inline auto exp10(wp_float3 x) noexcept { return wp_float3(exp10f(x.x()), exp10f(x.y()), exp10f(x.z())); }
CUDA_CALLABLE_DEVICE inline auto exp10(wp_float4 x) noexcept { return wp_float4(exp10f(x.x()), exp10f(x.y()), exp10f(x.z()), exp10f(x.w())); }

CUDA_CALLABLE_DEVICE inline auto log(wp_float2 x) noexcept { return wp_float2(logf(x.x()), logf(x.y())); }
CUDA_CALLABLE_DEVICE inline auto log(wp_float3 x) noexcept { return wp_float3(logf(x.x()), logf(x.y()), logf(x.z())); }
CUDA_CALLABLE_DEVICE inline auto log(wp_float4 x) noexcept { return wp_float4(logf(x.x()), logf(x.y()), logf(x.z()), logf(x.w())); }

CUDA_CALLABLE_DEVICE inline auto log2(wp_float2 x) noexcept { return wp_float2(log2f(x.x()), log2f(x.y())); }
CUDA_CALLABLE_DEVICE inline auto log2(wp_float3 x) noexcept { return wp_float3(log2f(x.x()), log2f(x.y()), log2f(x.z())); }
CUDA_CALLABLE_DEVICE inline auto log2(wp_float4 x) noexcept { return wp_float4(log2f(x.x()), log2f(x.y()), log2f(x.z()), log2f(x.w())); }

CUDA_CALLABLE_DEVICE inline auto log10(wp_float2 x) noexcept { return wp_float2(log10f(x.x()), log10f(x.y())); }
CUDA_CALLABLE_DEVICE inline auto log10(wp_float3 x) noexcept { return wp_float3(log10f(x.x()), log10f(x.y()), log10f(x.z())); }
CUDA_CALLABLE_DEVICE inline auto log10(wp_float4 x) noexcept { return wp_float4(log10f(x.x()), log10f(x.y()), log10f(x.z()), log10f(x.w())); }

CUDA_CALLABLE_DEVICE inline auto pow(wp_float2 x, wp_float2 a) noexcept { return wp_float2(powf(x.x(), a.x()), powf(x.y(), a.y())); }
CUDA_CALLABLE_DEVICE inline auto pow(wp_float3 x, wp_float3 a) noexcept { return wp_float3(powf(x.x(), a.x()), powf(x.y(), a.y()), powf(x.z(), a.z())); }
CUDA_CALLABLE_DEVICE inline auto pow(wp_float4 x, wp_float4 a) noexcept { return wp_float4(powf(x.x(), a.x()), powf(x.y(), a.y()), powf(x.z(), a.z()), powf(x.w(), a.w())); }

CUDA_CALLABLE_DEVICE inline auto sqrt(wp_float2 x) noexcept { return wp_float2(sqrtf(x.x()), sqrtf(x.y())); }
CUDA_CALLABLE_DEVICE inline auto sqrt(wp_float3 x) noexcept { return wp_float3(sqrtf(x.x()), sqrtf(x.y()), sqrtf(x.z())); }
CUDA_CALLABLE_DEVICE inline auto sqrt(wp_float4 x) noexcept { return wp_float4(sqrtf(x.x()), sqrtf(x.y()), sqrtf(x.z()), sqrtf(x.w())); }

CUDA_CALLABLE_DEVICE inline auto ceil(wp_float2 x) noexcept { return wp_float2(ceilf(x.x()), ceilf(x.y())); }
CUDA_CALLABLE_DEVICE inline auto ceil(wp_float3 x) noexcept { return wp_float3(ceilf(x.x()), ceilf(x.y()), ceilf(x.z())); }
CUDA_CALLABLE_DEVICE inline auto ceil(wp_float4 x) noexcept { return wp_float4(ceilf(x.x()), ceilf(x.y()), ceilf(x.z()), ceilf(x.w())); }

CUDA_CALLABLE_DEVICE inline auto floor(wp_float2 x) noexcept { return wp_float2(floorf(x.x()), floorf(x.y())); }
CUDA_CALLABLE_DEVICE inline auto floor(wp_float3 x) noexcept { return wp_float3(floorf(x.x()), floorf(x.y()), floorf(x.z())); }
CUDA_CALLABLE_DEVICE inline auto floor(wp_float4 x) noexcept { return wp_float4(floorf(x.x()), floorf(x.y()), floorf(x.z()), floorf(x.w())); }

CUDA_CALLABLE_DEVICE inline auto trunc(wp_float2 x) noexcept { return wp_float2(truncf(x.x()), truncf(x.y())); }
CUDA_CALLABLE_DEVICE inline auto trunc(wp_float3 x) noexcept { return wp_float3(truncf(x.x()), truncf(x.y()), truncf(x.z())); }
CUDA_CALLABLE_DEVICE inline auto trunc(wp_float4 x) noexcept { return wp_float4(truncf(x.x()), truncf(x.y()), truncf(x.z()), truncf(x.w())); }

CUDA_CALLABLE_DEVICE inline auto round(wp_float2 x) noexcept { return wp_float2(roundf(x.x()), roundf(x.y())); }
CUDA_CALLABLE_DEVICE inline auto round(wp_float3 x) noexcept { return wp_float3(roundf(x.x()), roundf(x.y()), roundf(x.z())); }
CUDA_CALLABLE_DEVICE inline auto round(wp_float4 x) noexcept { return wp_float4(roundf(x.x()), roundf(x.y()), roundf(x.z()), roundf(x.w())); }

CUDA_CALLABLE_DEVICE inline auto fma(wp_float x, wp_float y, wp_float z) noexcept { return fmaf(x, y, z); }
CUDA_CALLABLE_DEVICE inline auto fma(wp_float2 x, wp_float2 y, wp_float2 z) noexcept { return wp_float2(fmaf(x.x(), y.x(), z.x()), fmaf(x.y(), y.y(), z.y())); }
CUDA_CALLABLE_DEVICE inline auto fma(wp_float3 x, wp_float3 y, wp_float3 z) noexcept { return wp_float3(fmaf(x.x(), y.x(), z.x()), fmaf(x.y(), y.y(), z.y()), fmaf(x.z(), y.z(), z.z())); }
CUDA_CALLABLE_DEVICE inline auto fma(wp_float4 x, wp_float4 y, wp_float4 z) noexcept { return wp_float4(fmaf(x.x(), y.x(), z.x()), fmaf(x.y(), y.y(), z.y()), fmaf(x.z(), y.z(), z.z()), fmaf(x.w(), y.w(), z.w())); }

CUDA_CALLABLE_DEVICE inline auto copysign(wp_float x, wp_float y) noexcept { return copysignf(x, y); }
CUDA_CALLABLE_DEVICE inline auto copysign(wp_float2 x, wp_float2 y) noexcept { return wp_float2(copysignf(x.x(), y.x()), copysignf(x.y(), y.y())); }
CUDA_CALLABLE_DEVICE inline auto copysign(wp_float3 x, wp_float3 y) noexcept { return wp_float3(copysignf(x.x(), y.x()), copysignf(x.y(), y.y()), copysignf(x.z(), y.z())); }
CUDA_CALLABLE_DEVICE inline auto copysign(wp_float4 x, wp_float4 y) noexcept { return wp_float4(copysignf(x.x(), y.x()), copysignf(x.y(), y.y()), copysignf(x.z(), y.z()), copysignf(x.w(), y.w())); }

CUDA_CALLABLE_DEVICE inline auto reduce_sum(wp_short2 v) noexcept { return wp_short(v.x() + v.y()); }
CUDA_CALLABLE_DEVICE inline auto reduce_prod(wp_short2 v) noexcept { return wp_short(v.x() * v.y()); }
CUDA_CALLABLE_DEVICE inline auto reduce_min(wp_short2 v) noexcept { return wp_short(min(v.x(), v.y())); }
CUDA_CALLABLE_DEVICE inline auto reduce_max(wp_short2 v) noexcept { return wp_short(max(v.x(), v.y())); }
CUDA_CALLABLE_DEVICE inline auto reduce_sum(wp_short3 v) noexcept { return wp_short(v.x() + v.y() + v.z()); }
CUDA_CALLABLE_DEVICE inline auto reduce_prod(wp_short3 v) noexcept { return wp_short(v.x() * v.y() * v.z()); }
CUDA_CALLABLE_DEVICE inline auto reduce_min(wp_short3 v) noexcept { return wp_short(min(v.x(), min(v.y(), v.z()))); }
CUDA_CALLABLE_DEVICE inline auto reduce_max(wp_short3 v) noexcept { return wp_short(max(v.x(), max(v.y(), v.z()))); }
CUDA_CALLABLE_DEVICE inline auto reduce_sum(wp_short4 v) noexcept { return wp_short(v.x() + v.y() + v.z() + v.w()); }
CUDA_CALLABLE_DEVICE inline auto reduce_prod(wp_short4 v) noexcept { return wp_short(v.x() * v.y() * v.z() * v.w()); }
CUDA_CALLABLE_DEVICE inline auto reduce_min(wp_short4 v) noexcept { return wp_short(min(v.x(), min(v.y(), min(v.z(), v.w())))); }
CUDA_CALLABLE_DEVICE inline auto reduce_max(wp_short4 v) noexcept { return wp_short(max(v.x(), max(v.y(), max(v.z(), v.w())))); }
CUDA_CALLABLE_DEVICE inline auto reduce_sum(wp_ushort2 v) noexcept { return wp_ushort(v.x() + v.y()); }
CUDA_CALLABLE_DEVICE inline auto reduce_prod(wp_ushort2 v) noexcept { return wp_ushort(v.x() * v.y()); }
CUDA_CALLABLE_DEVICE inline auto reduce_min(wp_ushort2 v) noexcept { return wp_ushort(min(v.x(), v.y())); }
CUDA_CALLABLE_DEVICE inline auto reduce_max(wp_ushort2 v) noexcept { return wp_ushort(max(v.x(), v.y())); }
CUDA_CALLABLE_DEVICE inline auto reduce_sum(wp_ushort3 v) noexcept { return wp_ushort(v.x() + v.y() + v.z()); }
CUDA_CALLABLE_DEVICE inline auto reduce_prod(wp_ushort3 v) noexcept { return wp_ushort(v.x() * v.y() * v.z()); }
CUDA_CALLABLE_DEVICE inline auto reduce_min(wp_ushort3 v) noexcept { return wp_ushort(min(v.x(), min(v.y(), v.z()))); }
CUDA_CALLABLE_DEVICE inline auto reduce_max(wp_ushort3 v) noexcept { return wp_ushort(max(v.x(), max(v.y(), v.z()))); }
CUDA_CALLABLE_DEVICE inline auto reduce_sum(wp_ushort4 v) noexcept { return wp_ushort(v.x() + v.y() + v.z() + v.w()); }
CUDA_CALLABLE_DEVICE inline auto reduce_prod(wp_ushort4 v) noexcept { return wp_ushort(v.x() * v.y() * v.z() * v.w()); }
CUDA_CALLABLE_DEVICE inline auto reduce_min(wp_ushort4 v) noexcept { return wp_ushort(min(v.x(), min(v.y(), min(v.z(), v.w())))); }
CUDA_CALLABLE_DEVICE inline auto reduce_max(wp_ushort4 v) noexcept { return wp_ushort(max(v.x(), max(v.y(), max(v.z(), v.w())))); }
CUDA_CALLABLE_DEVICE inline auto reduce_sum(wp_int2 v) noexcept { return wp_int(v.x() + v.y()); }
CUDA_CALLABLE_DEVICE inline auto reduce_prod(wp_int2 v) noexcept { return wp_int(v.x() * v.y()); }
CUDA_CALLABLE_DEVICE inline auto reduce_min(wp_int2 v) noexcept { return wp_int(min(v.x(), v.y())); }
CUDA_CALLABLE_DEVICE inline auto reduce_max(wp_int2 v) noexcept { return wp_int(max(v.x(), v.y())); }
CUDA_CALLABLE_DEVICE inline auto reduce_sum(wp_int3 v) noexcept { return wp_int(v.x() + v.y() + v.z()); }
CUDA_CALLABLE_DEVICE inline auto reduce_prod(wp_int3 v) noexcept { return wp_int(v.x() * v.y() * v.z()); }
CUDA_CALLABLE_DEVICE inline auto reduce_min(wp_int3 v) noexcept { return wp_int(min(v.x(), min(v.y(), v.z()))); }
CUDA_CALLABLE_DEVICE inline auto reduce_max(wp_int3 v) noexcept { return wp_int(max(v.x(), max(v.y(), v.z()))); }
CUDA_CALLABLE_DEVICE inline auto reduce_sum(wp_int4 v) noexcept { return wp_int(v.x() + v.y() + v.z() + v.w()); }
CUDA_CALLABLE_DEVICE inline auto reduce_prod(wp_int4 v) noexcept { return wp_int(v.x() * v.y() * v.z() * v.w()); }
CUDA_CALLABLE_DEVICE inline auto reduce_min(wp_int4 v) noexcept { return wp_int(min(v.x(), min(v.y(), min(v.z(), v.w())))); }
CUDA_CALLABLE_DEVICE inline auto reduce_max(wp_int4 v) noexcept { return wp_int(max(v.x(), max(v.y(), max(v.z(), v.w())))); }
CUDA_CALLABLE_DEVICE inline auto reduce_sum(wp_uint2 v) noexcept { return wp_uint(v.x() + v.y()); }
CUDA_CALLABLE_DEVICE inline auto reduce_prod(wp_uint2 v) noexcept { return wp_uint(v.x() * v.y()); }
CUDA_CALLABLE_DEVICE inline auto reduce_min(wp_uint2 v) noexcept { return wp_uint(min(v.x(), v.y())); }
CUDA_CALLABLE_DEVICE inline auto reduce_max(wp_uint2 v) noexcept { return wp_uint(max(v.x(), v.y())); }
CUDA_CALLABLE_DEVICE inline auto reduce_sum(wp_uint3 v) noexcept { return wp_uint(v.x() + v.y() + v.z()); }
CUDA_CALLABLE_DEVICE inline auto reduce_prod(wp_uint3 v) noexcept { return wp_uint(v.x() * v.y() * v.z()); }
CUDA_CALLABLE_DEVICE inline auto reduce_min(wp_uint3 v) noexcept { return wp_uint(min(v.x(), min(v.y(), v.z()))); }
CUDA_CALLABLE_DEVICE inline auto reduce_max(wp_uint3 v) noexcept { return wp_uint(max(v.x(), max(v.y(), v.z()))); }
CUDA_CALLABLE_DEVICE inline auto reduce_sum(wp_uint4 v) noexcept { return wp_uint(v.x() + v.y() + v.z() + v.w()); }
CUDA_CALLABLE_DEVICE inline auto reduce_prod(wp_uint4 v) noexcept { return wp_uint(v.x() * v.y() * v.z() * v.w()); }
CUDA_CALLABLE_DEVICE inline auto reduce_min(wp_uint4 v) noexcept { return wp_uint(min(v.x(), min(v.y(), min(v.z(), v.w())))); }
CUDA_CALLABLE_DEVICE inline auto reduce_max(wp_uint4 v) noexcept { return wp_uint(max(v.x(), max(v.y(), max(v.z(), v.w())))); }
CUDA_CALLABLE_DEVICE inline auto reduce_sum(wp_float2 v) noexcept { return wp_float(v.x() + v.y()); }
CUDA_CALLABLE_DEVICE inline auto reduce_prod(wp_float2 v) noexcept { return wp_float(v.x() * v.y()); }
CUDA_CALLABLE_DEVICE inline auto reduce_min(wp_float2 v) noexcept { return wp_float(min(v.x(), v.y())); }
CUDA_CALLABLE_DEVICE inline auto reduce_max(wp_float2 v) noexcept { return wp_float(max(v.x(), v.y())); }
CUDA_CALLABLE_DEVICE inline auto reduce_sum(wp_float3 v) noexcept { return wp_float(v.x() + v.y() + v.z()); }
CUDA_CALLABLE_DEVICE inline auto reduce_prod(wp_float3 v) noexcept { return wp_float(v.x() * v.y() * v.z()); }
CUDA_CALLABLE_DEVICE inline auto reduce_min(wp_float3 v) noexcept { return wp_float(min(v.x(), min(v.y(), v.z()))); }
CUDA_CALLABLE_DEVICE inline auto reduce_max(wp_float3 v) noexcept { return wp_float(max(v.x(), max(v.y(), v.z()))); }
CUDA_CALLABLE_DEVICE inline auto reduce_sum(wp_float4 v) noexcept { return wp_float(v.x() + v.y() + v.z() + v.w()); }
CUDA_CALLABLE_DEVICE inline auto reduce_prod(wp_float4 v) noexcept { return wp_float(v.x() * v.y() * v.z() * v.w()); }
CUDA_CALLABLE_DEVICE inline auto reduce_min(wp_float4 v) noexcept { return wp_float(min(v.x(), min(v.y(), min(v.z(), v.w())))); }
CUDA_CALLABLE_DEVICE inline auto reduce_max(wp_float4 v) noexcept { return wp_float(max(v.x(), max(v.y(), max(v.z(), v.w())))); }
CUDA_CALLABLE_DEVICE inline auto min_impl(wp_short a, wp_short b) noexcept { return a < b ? a : b; }
CUDA_CALLABLE_DEVICE inline auto max_impl(wp_short a, wp_short b) noexcept { return a > b ? a : b; }
CUDA_CALLABLE_DEVICE inline auto min_impl(wp_ushort a, wp_ushort b) noexcept { return a < b ? a : b; }
CUDA_CALLABLE_DEVICE inline auto max_impl(wp_ushort a, wp_ushort b) noexcept { return a > b ? a : b; }
CUDA_CALLABLE_DEVICE inline auto min_impl(wp_int a, wp_int b) noexcept { return a < b ? a : b; }
CUDA_CALLABLE_DEVICE inline auto max_impl(wp_int a, wp_int b) noexcept { return a > b ? a : b; }
CUDA_CALLABLE_DEVICE inline auto min_impl(wp_uint a, wp_uint b) noexcept { return a < b ? a : b; }
CUDA_CALLABLE_DEVICE inline auto max_impl(wp_uint a, wp_uint b) noexcept { return a > b ? a : b; }
CUDA_CALLABLE_DEVICE inline auto min(wp_int2 a, wp_int2 b) noexcept { return wp_int2(min_impl(a.x(), b.x()), min_impl(a.y(), b.y())); }
CUDA_CALLABLE_DEVICE inline auto min(wp_int3 a, wp_int3 b) noexcept { return wp_int3(min_impl(a.x(), b.x()), min_impl(a.y(), b.y()), min_impl(a.z(), b.z())); }
CUDA_CALLABLE_DEVICE inline auto min(wp_int4 a, wp_int4 b) noexcept { return wp_int4(min_impl(a.x(), b.x()), min_impl(a.y(), b.y()), min_impl(a.z(), b.z()), min_impl(a.w(), b.w())); }
CUDA_CALLABLE_DEVICE inline auto min(wp_uint2 a, wp_uint2 b) noexcept { return wp_uint2(min_impl(a.x(), b.x()), min_impl(a.y(), b.y())); }
CUDA_CALLABLE_DEVICE inline auto min(wp_uint3 a, wp_uint3 b) noexcept { return wp_uint3(min_impl(a.x(), b.x()), min_impl(a.y(), b.y()), min_impl(a.z(), b.z())); }
CUDA_CALLABLE_DEVICE inline auto min(wp_uint4 a, wp_uint4 b) noexcept { return wp_uint4(min_impl(a.x(), b.x()), min_impl(a.y(), b.y()), min_impl(a.z(), b.z()), min_impl(a.w(), b.w())); }

CUDA_CALLABLE_DEVICE inline auto max(wp_int2 a, wp_int2 b) noexcept { return wp_int2(max_impl(a.x(), b.x()), max_impl(a.y(), b.y())); }
CUDA_CALLABLE_DEVICE inline auto max(wp_int3 a, wp_int3 b) noexcept { return wp_int3(max_impl(a.x(), b.x()), max_impl(a.y(), b.y()), max_impl(a.z(), b.z())); }
CUDA_CALLABLE_DEVICE inline auto max(wp_int4 a, wp_int4 b) noexcept { return wp_int4(max_impl(a.x(), b.x()), max_impl(a.y(), b.y()), max_impl(a.z(), b.z()), max_impl(a.w(), b.w())); }
CUDA_CALLABLE_DEVICE inline auto max(wp_uint2 a, wp_uint2 b) noexcept { return wp_uint2(max_impl(a.x(), b.x()), max_impl(a.y(), b.y())); }
CUDA_CALLABLE_DEVICE inline auto max(wp_uint3 a, wp_uint3 b) noexcept { return wp_uint3(max_impl(a.x(), b.x()), max_impl(a.y(), b.y()), max_impl(a.z(), b.z())); }
CUDA_CALLABLE_DEVICE inline auto max(wp_uint4 a, wp_uint4 b) noexcept { return wp_uint4(max_impl(a.x(), b.x()), max_impl(a.y(), b.y()), max_impl(a.z(), b.z()), max_impl(a.w(), b.w())); }

CUDA_CALLABLE_DEVICE inline auto clamp_impl(wp_short v, wp_short lo, wp_short hi) noexcept { return min(max(v, lo), hi); }
CUDA_CALLABLE_DEVICE inline auto clamp_impl(wp_ushort v, wp_ushort lo, wp_ushort hi) noexcept { return min(max(v, lo), hi); }
CUDA_CALLABLE_DEVICE inline auto clamp_impl(wp_int v, wp_int lo, wp_int hi) noexcept { return min(max(v, lo), hi); }
CUDA_CALLABLE_DEVICE inline auto clamp_impl(wp_uint v, wp_uint lo, wp_uint hi) noexcept { return min(max(v, lo), hi); }
CUDA_CALLABLE_DEVICE inline auto clamp_impl(wp_float v, wp_float lo, wp_float hi) noexcept { return min(max(v, lo), hi); }

CUDA_CALLABLE_DEVICE inline auto clamp(wp_int2 v, wp_int2 lo, wp_int2 hi) noexcept { return wp_int2(clamp_impl(v.x(), lo.x(), hi.x()), clamp_impl(v.y(), lo.y(), hi.y())); }
CUDA_CALLABLE_DEVICE inline auto clamp(wp_int3 v, wp_int3 lo, wp_int3 hi) noexcept { return wp_int3(clamp_impl(v.x(), lo.x(), hi.x()), clamp_impl(v.y(), lo.y(), hi.y()), clamp_impl(v.z(), lo.z(), hi.z())); }
CUDA_CALLABLE_DEVICE inline auto clamp(wp_int4 v, wp_int4 lo, wp_int4 hi) noexcept { return wp_int4(clamp_impl(v.x(), lo.x(), hi.x()), clamp_impl(v.y(), lo.y(), hi.y()), clamp_impl(v.z(), lo.z(), hi.z()), clamp_impl(v.w(), lo.w(), hi.w())); }
CUDA_CALLABLE_DEVICE inline auto clamp(wp_uint2 v, wp_uint2 lo, wp_uint2 hi) noexcept { return wp_uint2(clamp_impl(v.x(), lo.x(), hi.x()), clamp_impl(v.y(), lo.y(), hi.y())); }
CUDA_CALLABLE_DEVICE inline auto clamp(wp_uint3 v, wp_uint3 lo, wp_uint3 hi) noexcept { return wp_uint3(clamp_impl(v.x(), lo.x(), hi.x()), clamp_impl(v.y(), lo.y(), hi.y()), clamp_impl(v.z(), lo.z(), hi.z())); }
CUDA_CALLABLE_DEVICE inline auto clamp(wp_uint4 v, wp_uint4 lo, wp_uint4 hi) noexcept { return wp_uint4(clamp_impl(v.x(), lo.x(), hi.x()), clamp_impl(v.y(), lo.y(), hi.y()), clamp_impl(v.z(), lo.z(), hi.z()), clamp_impl(v.w(), lo.w(), hi.w())); }
CUDA_CALLABLE_DEVICE inline auto clamp(wp_float2 v, wp_float2 lo, wp_float2 hi) noexcept { return wp_float2(clamp_impl(v.x(), lo.x(), hi.x()), clamp_impl(v.y(), lo.y(), hi.y())); }
CUDA_CALLABLE_DEVICE inline auto clamp(wp_float3 v, wp_float3 lo, wp_float3 hi) noexcept { return wp_float3(clamp_impl(v.x(), lo.x(), hi.x()), clamp_impl(v.y(), lo.y(), hi.y()), clamp_impl(v.z(), lo.z(), hi.z())); }
CUDA_CALLABLE_DEVICE inline auto clamp(wp_float4 v, wp_float4 lo, wp_float4 hi) noexcept { return wp_float4(clamp_impl(v.x(), lo.x(), hi.x()), clamp_impl(v.y(), lo.y(), hi.y()), clamp_impl(v.z(), lo.z(), hi.z()), clamp_impl(v.w(), lo.w(), hi.w())); }

CUDA_CALLABLE_DEVICE inline auto lerp_impl(wp_float a, wp_float b, wp_float t) noexcept { return t * (b - a) + a; }
CUDA_CALLABLE_DEVICE inline auto lerp(wp_float a, wp_float b, wp_float t) noexcept { return lerp_impl(a, b, t); }
CUDA_CALLABLE_DEVICE inline auto lerp(wp_float2 a, wp_float2 b, wp_float2 t) noexcept { return wp_float2(lerp_impl(a.x(), b.x(), t.x()), lerp_impl(a.y(), b.y(), t.y())); }
CUDA_CALLABLE_DEVICE inline auto lerp(wp_float3 a, wp_float3 b, wp_float3 t) noexcept { return wp_float3(lerp_impl(a.x(), b.x(), t.x()), lerp_impl(a.y(), b.y(), t.y()), lerp_impl(a.z(), b.z(), t.z())); }
CUDA_CALLABLE_DEVICE inline auto lerp(wp_float4 a, wp_float4 b, wp_float4 t) noexcept { return wp_float4(lerp_impl(a.x(), b.x(), t.x()), lerp_impl(a.y(), b.y(), t.y()), lerp_impl(a.z(), b.z(), t.z()), lerp_impl(a.w(), b.w(), t.w())); }

CUDA_CALLABLE_DEVICE inline auto saturate(wp_float x) noexcept { return clamp(x, 0.0f, 1.0f); }
CUDA_CALLABLE_DEVICE inline auto saturate(wp_float2 x) noexcept { return clamp(x, wp_float2(0.0f), wp_float2(1.0f)); }
CUDA_CALLABLE_DEVICE inline auto saturate(wp_float3 x) noexcept { return clamp(x, wp_float3(0.0f), wp_float3(1.0f)); }
CUDA_CALLABLE_DEVICE inline auto saturate(wp_float4 x) noexcept { return clamp(x, wp_float4(0.0f), wp_float4(1.0f)); }

CUDA_CALLABLE_DEVICE inline auto degrees_impl(wp_float rad) noexcept { return rad * (180.0f * 0.318309886183790671537767526745028724f); }
CUDA_CALLABLE_DEVICE inline auto degrees(wp_float2 rad) noexcept { return wp_float2(degrees_impl(rad.x()), degrees_impl(rad.y())); }
CUDA_CALLABLE_DEVICE inline auto degrees(wp_float3 rad) noexcept { return wp_float3(degrees_impl(rad.x()), degrees_impl(rad.y()), degrees_impl(rad.z())); }
CUDA_CALLABLE_DEVICE inline auto degrees(wp_float4 rad) noexcept { return wp_float4(degrees_impl(rad.x()), degrees_impl(rad.y()), degrees_impl(rad.z()), degrees_impl(rad.w())); }

CUDA_CALLABLE_DEVICE inline auto radians_impl(wp_float deg) noexcept { return deg * (3.14159265358979323846264338327950288f / 180.0f); }
CUDA_CALLABLE_DEVICE inline auto radians(wp_float2 deg) noexcept { return wp_float2(radians_impl(deg.x()), radians_impl(deg.y())); }
CUDA_CALLABLE_DEVICE inline auto radians(wp_float3 deg) noexcept { return wp_float3(radians_impl(deg.x()), radians_impl(deg.y()), radians_impl(deg.z())); }
CUDA_CALLABLE_DEVICE inline auto radians(wp_float4 deg) noexcept { return wp_float4(radians_impl(deg.x()), radians_impl(deg.y()), radians_impl(deg.z()), radians_impl(deg.w())); }

CUDA_CALLABLE_DEVICE inline auto step_impl(wp_float edge, wp_float x) noexcept { return x < edge ? 0.0f : 1.0f; }
CUDA_CALLABLE_DEVICE inline auto step(wp_float edge, wp_float x) noexcept { return step_impl(edge, x); }
CUDA_CALLABLE_DEVICE inline auto step(wp_float2 edge, wp_float2 x) noexcept { return wp_float2(step_impl(edge.x(), x.x()), step_impl(edge.y(), x.y())); }
CUDA_CALLABLE_DEVICE inline auto step(wp_float3 edge, wp_float3 x) noexcept { return wp_float3(step_impl(edge.x(), x.x()), step_impl(edge.y(), x.y()), step_impl(edge.z(), x.z())); }
CUDA_CALLABLE_DEVICE inline auto step(wp_float4 edge, wp_float4 x) noexcept { return wp_float4(step_impl(edge.x(), x.x()), step_impl(edge.y(), x.y()), step_impl(edge.z(), x.z()), step_impl(edge.w(), x.w())); }

CUDA_CALLABLE_DEVICE inline auto smoothstep_impl(wp_float edge0, wp_float edge1, wp_float x) noexcept {
    auto t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.f - 2.f * t);
}
CUDA_CALLABLE_DEVICE inline auto smoothstep(wp_float2 edge0, wp_float2 edge1, wp_float2 x) noexcept { return wp_float2(smoothstep_impl(edge0.x(), edge1.x(), x.x()), smoothstep_impl(edge0.y(), edge1.y(), x.y())); }
CUDA_CALLABLE_DEVICE inline auto smoothstep(wp_float3 edge0, wp_float3 edge1, wp_float3 x) noexcept { return wp_float3(smoothstep_impl(edge0.x(), edge1.x(), x.x()), smoothstep_impl(edge0.y(), edge1.y(), x.y()), smoothstep_impl(edge0.z(), edge1.z(), x.z())); }
CUDA_CALLABLE_DEVICE inline auto smoothstep(wp_float4 edge0, wp_float4 edge1, wp_float4 x) noexcept { return wp_float4(smoothstep_impl(edge0.x(), edge1.x(), x.x()), smoothstep_impl(edge0.y(), edge1.y(), x.y()), smoothstep_impl(edge0.z(), edge1.z(), x.z()), smoothstep_impl(edge0.w(), edge1.w(), x.w())); }

CUDA_CALLABLE_DEVICE inline auto mod_impl(wp_float x, wp_float y) noexcept { return x - y * floor(x / y); }
CUDA_CALLABLE_DEVICE inline auto mod(wp_float2 x, wp_float2 y) noexcept { return wp_float2(mod_impl(x.x(), y.x()), mod_impl(x.y(), y.y())); }
CUDA_CALLABLE_DEVICE inline auto mod(wp_float3 x, wp_float3 y) noexcept { return wp_float3(mod_impl(x.x(), y.x()), mod_impl(x.y(), y.y()), mod_impl(x.z(), y.z())); }
CUDA_CALLABLE_DEVICE inline auto mod(wp_float4 x, wp_float4 y) noexcept { return wp_float4(mod_impl(x.x(), y.x()), mod_impl(x.y(), y.y()), mod_impl(x.z(), y.z()), mod_impl(x.w(), y.w())); }

CUDA_CALLABLE_DEVICE inline auto fmod(wp_float x, wp_float y) noexcept { return fmodf(x, y); }
CUDA_CALLABLE_DEVICE inline auto fmod(wp_float2 x, wp_float2 y) noexcept { return wp_float2(fmodf(x.x(), y.x()), fmodf(x.y(), y.y())); }
CUDA_CALLABLE_DEVICE inline auto fmod(wp_float3 x, wp_float3 y) noexcept { return wp_float3(fmodf(x.x(), y.x()), fmodf(x.y(), y.y()), fmodf(x.z(), y.z())); }
CUDA_CALLABLE_DEVICE inline auto fmod(wp_float4 x, wp_float4 y) noexcept { return wp_float4(fmodf(x.x(), y.x()), fmodf(x.y(), y.y()), fmodf(x.z(), y.z()), fmodf(x.w(), y.w())); }

CUDA_CALLABLE_DEVICE inline auto fract_impl(wp_float x) noexcept { return x - floor(x); }
CUDA_CALLABLE_DEVICE inline float fract(wp_float x) noexcept { return fract_impl(x); }
CUDA_CALLABLE_DEVICE inline auto fract(wp_float2 x) noexcept { return wp_float2(fract_impl(x.x()), fract_impl(x.y())); }
CUDA_CALLABLE_DEVICE inline auto fract(wp_float3 x) noexcept { return wp_float3(fract_impl(x.x()), fract_impl(x.y()), fract_impl(x.z())); }
CUDA_CALLABLE_DEVICE inline auto fract(wp_float4 x) noexcept { return wp_float4(fract_impl(x.x()), fract_impl(x.y()), fract_impl(x.z()), fract_impl(x.w())); }

CUDA_CALLABLE_DEVICE inline constexpr auto wp_cross(wp_float3 u, wp_float3 v) noexcept {
    return wp_float3(u.y() * v.z() - v.y() * u.z(),
                     u.z() * v.x() - v.z() * u.x(),
                     u.x() * v.y() - v.x() * u.y());
}

CUDA_CALLABLE_DEVICE inline auto dot(wp_float2 a, wp_float2 b) noexcept {
    return a.x() * b.x() + a.y() * b.y();
}
CUDA_CALLABLE_DEVICE inline auto dot(wp_float3 a, wp_float3 b) noexcept {
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}
CUDA_CALLABLE_DEVICE inline auto dot(wp_float4 a, wp_float4 b) noexcept {
    return a.x() * b.x() + a.y() * b.y() + a.z() * b.z() + a.w() * b.w();
}

CUDA_CALLABLE_DEVICE inline auto length(wp_float2 v) noexcept { return sqrtf(dot(v, v)); }
CUDA_CALLABLE_DEVICE inline auto length(wp_float3 v) noexcept { return sqrtf(dot(v, v)); }
CUDA_CALLABLE_DEVICE inline auto length(wp_float4 v) noexcept { return sqrtf(dot(v, v)); }

CUDA_CALLABLE_DEVICE inline auto length_squared(wp_float2 v) noexcept { return dot(v, v); }
CUDA_CALLABLE_DEVICE inline auto length_squared(wp_float3 v) noexcept { return dot(v, v); }
CUDA_CALLABLE_DEVICE inline auto length_squared(wp_float4 v) noexcept { return dot(v, v); }

CUDA_CALLABLE_DEVICE inline auto distance(wp_float2 a, wp_float2 b) noexcept { return length(a - b); }
CUDA_CALLABLE_DEVICE inline auto distance(wp_float3 a, wp_float3 b) noexcept { return length(a - b); }
CUDA_CALLABLE_DEVICE inline auto distance(wp_float4 a, wp_float4 b) noexcept { return length(a - b); }

CUDA_CALLABLE_DEVICE inline auto distance_squared(wp_float2 a, wp_float2 b) noexcept { return length_squared(a - b); }
CUDA_CALLABLE_DEVICE inline auto distance_squared(wp_float3 a, wp_float3 b) noexcept { return length_squared(a - b); }
CUDA_CALLABLE_DEVICE inline auto distance_squared(wp_float4 a, wp_float4 b) noexcept { return length_squared(a - b); }

CUDA_CALLABLE_DEVICE inline auto faceforward(wp_float3 n, wp_float3 i, wp_float3 n_ref) noexcept { return dot(n_ref, i) < 0.0f ? n : -n; }

}// namespace wp
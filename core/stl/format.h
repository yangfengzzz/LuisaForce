//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <type_traits>

#include <spdlog/fmt/fmt.h>
#include "core/basic_types.h"
#include "core/stl/string.h"

namespace luisa {
[[nodiscard]] inline auto hash_to_string(uint64_t hash) noexcept {
    return fmt::format(FMT_STRING("{:016X}"), hash);
}
};// namespace luisa

namespace fmt {

template<typename T, size_t N>
struct formatter<luisa::Vector<T, N>> {
    constexpr auto parse(format_parse_context &ctx) -> decltype(ctx.begin()) {
        return ctx.end();
    }
    template<typename FormatContext>
    auto format(const luisa::Vector<T, N> &v, FormatContext &ctx) -> decltype(ctx.out()) {
        using namespace std::string_view_literals;
        using luisa::uint;
        using luisa::ushort;
        using luisa::slong;
        using luisa::ulong;
        using luisa::half;
        constexpr auto type_name =
            std::is_same_v<T, bool>   ? "bool"sv :
            std::is_same_v<T, short>  ? "short"sv :
            std::is_same_v<T, ushort> ? "ushort"sv :
            std::is_same_v<T, int>    ? "int"sv :
            std::is_same_v<T, uint>   ? "uint"sv :
            std::is_same_v<T, slong>  ? "slong"sv :
            std::is_same_v<T, ulong>  ? "ulong"sv :
            std::is_same_v<T, half>   ? "half"sv :
            std::is_same_v<T, float>  ? "float"sv :
            std::is_same_v<T, double> ? "double"sv :
                                        "unknown"sv;
        if constexpr (N == 2u) {
            return fmt::format_to(
                ctx.out(),
                FMT_STRING("{}2({}, {})"),
                type_name, v.x, v.y);
        } else if constexpr (N == 3u) {
            return fmt::format_to(
                ctx.out(),
                FMT_STRING("{}3({}, {}, {})"),
                type_name, v.x, v.y, v.z);
        } else if constexpr (N == 4u) {
            return fmt::format_to(
                ctx.out(),
                FMT_STRING("{}4({}, {}, {}, {})"),
                type_name, v.x, v.y, v.z, v.w);
        } else {
            static_assert(luisa::always_false_v<T>);
        }
    }
};

template<size_t N>
struct formatter<luisa::Matrix<N>> {
    constexpr auto parse(format_parse_context &ctx) -> decltype(ctx.begin()) {
        return ctx.end();
    }
    template<typename FormatContext>
    auto format(const luisa::Matrix<N> &m, FormatContext &ctx) -> decltype(ctx.out()) {
        if constexpr (N == 2u) {
            return fmt::format_to(
                ctx.out(),
                FMT_STRING("float2x2("
                           "cols[0] = ({}, {}), "
                           "cols[1] = ({}, {}))"),
                m[0].x, m[0].y,
                m[1].x, m[1].y);
        } else if constexpr (N == 3u) {
            return fmt::format_to(
                ctx.out(),
                FMT_STRING("float3x3("
                           "cols[0] = ({}, {}, {}), "
                           "cols[1] = ({}, {}, {}), "
                           "cols[2] = ({}, {}, {}))"),
                m[0].x, m[0].y, m[0].z,
                m[1].x, m[1].y, m[1].z,
                m[2].x, m[2].y, m[2].z);
        } else if constexpr (N == 4u) {
            return fmt::format_to(
                ctx.out(),
                FMT_STRING("float4x4("
                           "cols[0] = ({}, {}, {}, {}), "
                           "cols[1] = ({}, {}, {}, {}), "
                           "cols[2] = ({}, {}, {}, {}), "
                           "cols[3] = ({}, {}, {}, {}))"),
                m[0].x, m[0].y, m[0].z, m[0].w,
                m[1].x, m[1].y, m[1].z, m[1].w,
                m[2].x, m[2].y, m[2].z, m[2].w,
                m[3].x, m[3].y, m[3].z, m[3].w);
        } else {
            static_assert(luisa::always_false_v<luisa::Matrix<N>>);
        }
    }
};

template<typename T, size_t N>
struct formatter<std::array<T, N>> {
    constexpr auto parse(format_parse_context &ctx) -> decltype(ctx.begin()) {
        return ctx.end();
    }
    template<typename FormatContext>
    auto format(const std::array<T, N> &a, FormatContext &ctx) -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), FMT_STRING("[{}]"), fmt::join(a, ", "));
    }
};

}// namespace fmt

namespace luisa {

template<typename T, size_t N>
[[nodiscard]] auto to_string(Vector<T, N> v) noexcept {
    return fmt::format(FMT_STRING("({})"), v);
}

template<size_t N>
[[nodiscard]] auto to_string(Matrix<N> m) noexcept {
    return fmt::format(FMT_STRING("({})"), m);
}

template<typename T, size_t N>
[[nodiscard]] auto to_string(std::array<T, N> a) noexcept {
    return fmt::format(FMT_STRING("({})"), a);
}

}// namespace luisa

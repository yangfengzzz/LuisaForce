//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <array>
#include <type_traits>

#include "core/stl/hash_fwd.h"
#include "core/stl/string.h"
#include "core/basic_traits.h"

namespace luisa {

template<typename T>
    requires std::disjunction_v<std::is_arithmetic<T>,
                                std::is_pointer<T>,
                                std::is_enum<T>,
                                std::is_same<T, half>>
struct hash<T> {
    using is_avalaunching = void;
    [[nodiscard]] constexpr uint64_t operator()(T value, uint64_t seed = hash64_default_seed) const noexcept {
        return hash64(&value, sizeof(T), seed);
    }
};

template<typename T>
    requires requires(const T v) { v.hash(); }
struct hash<T> {
    using is_avalaunching = void;
    [[nodiscard]] constexpr uint64_t operator()(const T &value) const noexcept {
        return value.hash();
    }
};

template<typename T>
[[nodiscard]] inline uint64_t hash_value(T &&value) noexcept {
    return hash<std::remove_cvref_t<T>>{}(std::forward<T>(value));
}

template<typename T>
[[nodiscard]] inline uint64_t hash_value(T &&value, uint64_t seed) noexcept {
    return hash<std::remove_cvref_t<T>>{}(std::forward<T>(value), seed);
}

[[nodiscard]] inline uint64_t hash_combine(std::initializer_list<uint64_t> values,
                                           uint64_t seed = hash64_default_seed) noexcept {
    return luisa::hash64(std::data(values), std::size(values) * sizeof(uint64_t), seed);
}

[[nodiscard]] inline uint64_t hash_combine(luisa::span<const uint64_t> values,
                                           uint64_t seed = hash64_default_seed) noexcept {
    return luisa::hash64(values.data(), values.size_bytes(), seed);
}

class LC_CORE_API Hash128 {

private:
    std::array<uint8_t, 16u> _data{};

public:
    Hash128() noexcept = default;
    explicit Hash128(span<std::uint8_t> data) noexcept;
    explicit Hash128(string_view s) noexcept;
    [[nodiscard]] auto data() noexcept { return luisa::span{_data}; }
    [[nodiscard]] auto data() const noexcept { return luisa::span{_data}; }
    [[nodiscard]] luisa::string to_string() const noexcept;
    [[nodiscard]] auto operator==(const Hash128 &rhs) const noexcept { return _data == rhs._data; }
};

}// namespace luisa

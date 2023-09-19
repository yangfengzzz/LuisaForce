//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/basic_types.h"
#include "core/stl/string.h"

namespace luisa::compute {

class StringScratch {

private:
    luisa::string _buffer;

public:
    explicit StringScratch(size_t reserved_size) noexcept;
    StringScratch() noexcept;
    StringScratch &operator<<(std::string_view s) noexcept;
    StringScratch &operator<<(const char *s) noexcept;
    StringScratch &operator<<(const std::string &s) noexcept;
    StringScratch &operator<<(bool x) noexcept;
    StringScratch &operator<<(float x) noexcept;
    StringScratch &operator<<(double x) noexcept;
    StringScratch &operator<<(int x) noexcept;
    StringScratch &operator<<(uint x) noexcept;
    StringScratch &operator<<(size_t x) noexcept;
    [[nodiscard]] const luisa::string &string() const noexcept;
    [[nodiscard]] luisa::string_view string_view() const noexcept;
    [[nodiscard]] const char *c_str() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] size_t size() const noexcept;
    void pop_back() noexcept;
    void clear() noexcept;
    [[nodiscard]] char back() const noexcept;
};

}// namespace luisa::compute

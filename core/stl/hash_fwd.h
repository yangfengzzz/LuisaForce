//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <cstddef>
#include <cstdint>

#include "core/dll_export.h"

namespace luisa {

static constexpr uint64_t hash64_default_seed = (1ull << 61ull) - 1ull;// (2^61 - 1) is a prime

[[nodiscard]] LC_CORE_API uint64_t hash64(const void *ptr, size_t size, uint64_t seed) noexcept;

template<typename T>
struct hash {};

}// namespace luisa

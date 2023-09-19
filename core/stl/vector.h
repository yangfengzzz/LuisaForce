//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <EASTL/vector.h>
#include <EASTL/fixed_vector.h>
#include <EASTL/bitvector.h>

namespace luisa {

template<typename T>
using vector = eastl::vector<T>;

template<typename T, size_t node_count, bool allow_overflow = true>
using fixed_vector = eastl::fixed_vector<T, node_count, allow_overflow>;

using bitvector = eastl::bitvector<>;

}// namespace luisa

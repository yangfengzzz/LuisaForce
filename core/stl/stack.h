//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <EASTL/stack.h>
#include "core/stl/vector.h"

namespace luisa {

template<typename T, typename Container = luisa::vector<T>>
using stack = eastl::stack<T, Container>;

}// namespace luisa

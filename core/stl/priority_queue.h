//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <EASTL/priority_queue.h>

#include "core/stl/vector.h"
#include "core/stl/functional.h"

namespace luisa {

template<typename T,
         typename Container = vector<T>,
         typename Compare = less<>>
using priority_queue = eastl::priority_queue<T, Container, Compare>;

}// namespace luisa

//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <EASTL/queue.h>

#include "core/stl/deque.h"

namespace luisa {

template<typename T, typename Container = luisa::deque<T>>
using queue = eastl::queue<T, Container>;

}// namespace luisa


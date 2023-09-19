//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <EASTL/bonus/ring_buffer.h>
#include <EASTL/bonus/fixed_ring_buffer.h>

namespace luisa {

template<typename T>
using ring_buffer = eastl::ring_buffer<T>;

template<typename T, size_t N>
using fixed_ring_buffer = eastl::fixed_ring_buffer<T, N>;

}// namespace luisa

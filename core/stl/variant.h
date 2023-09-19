//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <EASTL/variant.h>

namespace luisa {

using eastl::get;
using eastl::get_if;
using eastl::holds_alternative;
using eastl::monostate;
using eastl::variant;
using eastl::variant_alternative_t;
using eastl::variant_size_v;
using eastl::visit;

}// namespace luisa


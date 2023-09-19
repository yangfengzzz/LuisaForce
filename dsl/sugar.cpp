//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/stl/format.h"
#include "dsl/sugar.h"

namespace luisa::compute::dsl_detail {
std::string format_source_location(const char *file, int line) noexcept {
    return fmt::format("{}:{}", file, line);
}
}// namespace luisa::compute::dsl_detail

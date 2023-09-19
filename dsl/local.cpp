//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/logging.h"
#include "dsl/local.h"

namespace luisa::compute::detail {

void local_array_error_sizes_missmatch(size_t lhs, size_t rhs) noexcept {
    LUISA_ERROR_WITH_LOCATION(
        "Incompatible sizes ({} and {}).", lhs, rhs);
}

}// namespace luisa::compute::detail

//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/logging.h"
#include "core/pool.h"

namespace luisa {

void LC_CORE_API detail::memory_pool_check_memory_leak(size_t expected, size_t actual) noexcept {
    if (expected != actual) [[unlikely]] {
        LUISA_WARNING_WITH_LOCATION(
            "Leaks detected in pool: "
            "expected {} objects but got {}.",
            expected, actual);
    }
}

}// namespace luisa

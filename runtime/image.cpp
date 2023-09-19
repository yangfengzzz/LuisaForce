//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/logging.h"
#include "runtime/image.h"

namespace luisa::compute ::detail {

LC_RUNTIME_API void error_image_invalid_mip_levels(size_t level, size_t mip) noexcept {
    LUISA_ERROR_WITH_LOCATION(
        "Invalid mipmap level {} for image with {} levels.",
        level, mip);
}
LC_RUNTIME_API void image_size_zero_error() noexcept {
    LUISA_ERROR_WITH_LOCATION("Image size must be non-zero.");
}
LC_RUNTIME_API void volume_size_zero_error() noexcept {
    LUISA_ERROR_WITH_LOCATION("Volume size must be non-zero.");
}
}// namespace luisa::compute::detail

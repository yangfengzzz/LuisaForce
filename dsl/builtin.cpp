//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/basic_types.h"
#include "core/logging.h"

namespace luisa::compute::detail {

LC_DSL_API void validate_block_size(uint x, uint y, uint z) noexcept {
    auto size = make_uint3(x, y, z);
    LUISA_ASSERT(all(size >= 1u && size <= 1024u),
                 "Invalid block size ({}, {}, {}). "
                 "Block size must be in range [1, 1024].",
                 x, y, z);
    LUISA_ASSERT((x * y * z) % 32u == 0u,
                 "Invalid block size ({}, {}, {}). "
                 "Threads per block must be a multiple of 32.",
                 x, y, z);
}

}// namespace luisa::compute::detail

//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <cstdint>
#include "core/stl/string.h"

namespace luisa::compute {

enum struct CustomCommandUUID : uint32_t {
    CUSTOM_DISPATCH_EXT_BEGIN = 0x0000u,
    CUSTOM_DISPATCH = CUSTOM_DISPATCH_EXT_BEGIN,
    PERFORMANCE_DISPATCH,

    DSTORAGE_EXT_BEGIN = 0x0200u,
    DSTORAGE_READ = DSTORAGE_EXT_BEGIN,

    CUDA_CUSTOM_COMMAND_BEGIN = 0x0400u,
    CUDA_LCUB_COMMAND = CUDA_CUSTOM_COMMAND_BEGIN,

    REGISTERED_END = 0xffffu,
};

}// namespace luisa::compute

namespace luisa {

[[nodiscard]] inline luisa::string to_string(compute::CustomCommandUUID uuid) noexcept {
    switch (uuid) {
        case compute::CustomCommandUUID::CUSTOM_DISPATCH: return "CUSTOM_DISPATCH";
        case compute::CustomCommandUUID::DSTORAGE_READ: return "DSTORAGE_READ";
        case compute::CustomCommandUUID::CUDA_LCUB_COMMAND: return "CUDA_LCUB_COMMAND";
        default: break;
    }
    return "UNKNOWN";
}

}// namespace luisa

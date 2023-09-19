//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/logging.h"
#include "runtime/device.h"

namespace luisa::compute {

void Device::_check_no_implicit_binding(Function func, luisa::string_view shader_path) noexcept {
#ifndef NDEBUG
    for (auto &&b : func.bound_arguments()) {
        if (!holds_alternative<monostate>(b)) {
            LUISA_ERROR("Kernel {} with resource "
                        "bindings cannot be saved!",
                        shader_path);
        }
    }
#endif
}

}// namespace luisa::compute

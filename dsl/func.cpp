//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/logging.h"
#include "dsl/func.h"

namespace luisa::compute::detail {

void CallableInvoke::_error_too_many_arguments() noexcept {
    LUISA_ERROR_WITH_LOCATION("Too many arguments for callable.");
}

}// namespace luisa::compute::detail

namespace luisa::compute::detail {
luisa::shared_ptr<const FunctionBuilder>
transform_function(Function function) noexcept {
    return function.shared_builder();
}

}// namespace luisa::compute::detail

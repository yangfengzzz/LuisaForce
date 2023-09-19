//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/dll_export.h"
#include "core/stl/string.h"
#include "ast/function.h"

namespace luisa::compute {
[[nodiscard]] LC_AST_API luisa::string to_json(Function function) noexcept;
}// namespace luisa::compute

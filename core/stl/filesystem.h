//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <filesystem>
#include "core/stl/string.h"

namespace luisa {

namespace filesystem = std::filesystem;
[[nodiscard]] LC_CORE_API luisa::string to_string(const luisa::filesystem::path &path);

}// namespace luisa

//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/stl/filesystem.h"

namespace luisa {

LC_CORE_API luisa::string to_string(const luisa::filesystem::path &path) {
    return path.string<char, std::char_traits<char>, luisa::allocator<char>>();
}

}// namespace luisa


//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/stl/hash.h"
#include "ast/variable.h"

namespace luisa::compute {

uint64_t Variable::hash() const noexcept {
    using namespace std::string_view_literals;
    static auto seed = hash_value("__hash_variable"sv);
    auto u0 = static_cast<uint64_t>(_uid);
    auto u1 = static_cast<uint64_t>(_tag);
    return hash_combine({u0 | (u1 << 32u), _type->hash()}, seed);
}

}// namespace luisa::compute

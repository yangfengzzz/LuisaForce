//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/stl/memory.h"
#include "core/stl/vector.h"
#include "core/stl/functional.h"
#include "core/stl/hash.h"

#ifndef LUISA_COMPUTE_USE_DENSE_MAP
#define LUISA_COMPUTE_USE_DENSE_MAP 1
#endif

#if LUISA_COMPUTE_USE_DENSE_MAP
#include "core/stl/unordered_dense.h"
#else
#include <unordered_map>
#include <unordered_set>
#endif

namespace luisa {

#ifdef LUISA_COMPUTE_USE_DENSE_MAP

template<typename K, typename V,
         typename Hash = luisa::hash<K>,
         typename Eq = std::equal_to<>>
using unordered_map = ankerl::unordered_dense::map<
    K, V, Hash, Eq,
    luisa::allocator<std::pair<K, V>>,
    luisa::vector<std::pair<K, V>>>;

template<typename K,
         typename Hash = luisa::hash<K>,
         typename Eq = std::equal_to<>>
using unordered_set = ankerl::unordered_dense::set<
    K, Hash, Eq, luisa::allocator<K>, luisa::vector<K>>;

#else

template<typename K, typename V,
         typename Hash = hash<K>,
         typename Eq = equal_to<>>
using unordered_map = std::unordered_map<
    K, V, Hash, Eq, allocator<std::pair<const K, V>>>;

template<typename K,
         typename Hash = hash<K>,
         typename Eq = equal_to<>>
using unordered_set = std::unordered_set<
    K, Hash, Eq, allocator<K>>;

#endif

}// namespace luisa

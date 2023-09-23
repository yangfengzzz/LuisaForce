//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace luisa::compute::cuda::dcub {

template<typename _Key, typename _Value>
struct KeyValuePair {
    typedef _Key Key;    ///< Key data type
    typedef _Value Value;///< Value data type

    Key key;    ///< Item key
    Value value;///< Item value

    /// Constructor
    KeyValuePair() {}

    /// Constructor
    KeyValuePair(Key const &key, Value const &value) : key(key), value(value) {}
};

struct Equality {};

struct Max {};

struct Min {};

enum class BinaryOperator {
    Max,
    Min,
};
}// namespace luisa::compute::cuda::dcub
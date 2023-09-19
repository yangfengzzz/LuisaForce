//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <cstdint>

namespace luisa::compute {

struct Argument {
    enum struct Tag {
        BUFFER,
        TEXTURE,
        UNIFORM,
        BINDLESS_ARRAY
    };

    struct Buffer {
        uint64_t handle;
        size_t offset;
        size_t size;
    };

    struct Texture {
        uint64_t handle;
        uint32_t level;
    };

    struct Uniform {
        size_t offset;
        size_t size;
    };

    struct BindlessArray {
        uint64_t handle;
    };

    Tag tag;
    union {
        Buffer buffer;
        Texture texture;
        Uniform uniform;
        BindlessArray bindless_array;
    };
};

}// namespace luisa::compute

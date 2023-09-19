//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/binary_buffer.h"
#include <bit>
namespace luisa {

void BinaryBuffer::_write_bytes(const void *data, size_t size, size_t alignment) noexcept {
    auto offset = align(_bytes.size(), alignment);
    auto size_after_write = offset + size;
    auto required_capacity = std::bit_ceil(size_after_write);
    _bytes.reserve(required_capacity);
    auto ptr = _bytes.data() + offset;
    _bytes.resize(size_after_write);
    memcpy(ptr, data, size);
}

}// namespace luisa

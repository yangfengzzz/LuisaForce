//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <cstdio>

#include "core/stl/memory.h"
#include "core/stl/string.h"
#include "core/stl/filesystem.h"

namespace luisa {

class BinaryStream {

public:
    [[nodiscard]] virtual size_t length() const noexcept = 0;
    [[nodiscard]] virtual size_t pos() const noexcept = 0;
    virtual void read(luisa::span<std::byte> dst) noexcept = 0;
    virtual ~BinaryStream() noexcept = default;
};

class BinaryIO {

public:
    virtual ~BinaryIO() noexcept = default;
    [[nodiscard]] virtual luisa::unique_ptr<BinaryStream> read_shader_bytecode(luisa::string_view name) const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<BinaryStream> read_shader_cache(luisa::string_view name) const noexcept = 0;
    [[nodiscard]] virtual luisa::unique_ptr<BinaryStream> read_internal_shader(luisa::string_view name) const noexcept = 0;
    // returns the path of the written file (if stored on disk, otherwise returns empty path)
    [[nodiscard]] virtual luisa::filesystem::path write_shader_bytecode(luisa::string_view name, luisa::span<std::byte const> data) const noexcept = 0;
    [[nodiscard]] virtual luisa::filesystem::path write_shader_cache(luisa::string_view name, luisa::span<std::byte const> data) const noexcept = 0;
    [[nodiscard]] virtual luisa::filesystem::path write_internal_shader(luisa::string_view name, luisa::span<std::byte const> data) const noexcept = 0;
};

}// namespace luisa

//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <shared_mutex>

#include "core/binary_io.h"
#include "core/stl/filesystem.h"
#include "core/stl/unordered_map.h"
#include "runtime/context.h"

namespace luisa::compute {

class DefaultBinaryIO final : public BinaryIO {
private:
    Context _ctx;
    mutable std::mutex _global_mtx;
    std::filesystem::path _cache_dir;
    std::filesystem::path _data_dir;

private:
    luisa::unique_ptr<BinaryStream> _read(luisa::string const &file_path) const noexcept;
    void _write(luisa::string const &file_path, luisa::span<std::byte const> data) const noexcept;

public:
    explicit DefaultBinaryIO(Context &&ctx, void *ext = nullptr) noexcept;
    ~DefaultBinaryIO() noexcept override;
    luisa::unique_ptr<BinaryStream> read_shader_bytecode(luisa::string_view name) const noexcept override;
    luisa::unique_ptr<BinaryStream> read_shader_cache(luisa::string_view name) const noexcept override;
    luisa::unique_ptr<BinaryStream> read_internal_shader(luisa::string_view name) const noexcept override;
    luisa::filesystem::path write_shader_bytecode(luisa::string_view name, luisa::span<std::byte const> data) const noexcept override;
    luisa::filesystem::path write_shader_cache(luisa::string_view name, luisa::span<std::byte const> data) const noexcept override;
    luisa::filesystem::path write_internal_shader(luisa::string_view name, luisa::span<std::byte const> data) const noexcept override;
};

}// namespace luisa::compute

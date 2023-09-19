//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/stl/filesystem.h"
#include "core/logging.h"
#include "core/binary_file_stream.h"

#include "default_binary_io.h"

namespace luisa::compute {
class LockedBinaryFileStream : public BinaryStream {

private:
    BinaryFileStream _stream;
    DefaultBinaryIO const *_binary_io;

public:
    explicit LockedBinaryFileStream(DefaultBinaryIO const *binary_io, ::FILE *file, size_t length, const luisa::string &path) noexcept
        : _stream{file, length},
          _binary_io{binary_io} {}
    ~LockedBinaryFileStream() noexcept override = default;
    [[nodiscard]] size_t length() const noexcept override { return _stream.length(); }
    [[nodiscard]] size_t pos() const noexcept override { return _stream.pos(); }
    void read(luisa::span<std::byte> dst) noexcept override {
        _stream.read(dst);
    }
};

luisa::unique_ptr<BinaryStream> DefaultBinaryIO::_read(luisa::string const &file_path) const noexcept {
    auto file = std::fopen(file_path.c_str(), "rb");
    if (file) {
        auto length = BinaryFileStream::seek_len(file);
        if (length == 0) [[unlikely]] {
            return nullptr;
        }
        return luisa::make_unique<LockedBinaryFileStream>(this, file, length, file_path);
    } else {
        LUISA_VERBOSE("Read file {} failed.", file_path);
        return nullptr;
    }
}

void DefaultBinaryIO::_write(const luisa::string &file_path, luisa::span<std::byte const> data) const noexcept {
    auto folder = luisa::filesystem::path{file_path}.parent_path();
    std::error_code ec;
    luisa::filesystem::create_directories(folder, ec);
    if (ec) { LUISA_WARNING("Create directory {} failed.", folder.string()); }
    if (auto f = fopen(file_path.c_str(), "wb")) [[likely]] {
#ifdef _WIN32
#define LUISA_FWRITE _fwrite_nolock
#define LUISA_FCLOSE _fclose_nolock
#else
#define LUISA_FWRITE fwrite
#define LUISA_FCLOSE fclose
#endif
        LUISA_FWRITE(data.data(), data.size(), 1, f);
        LUISA_FCLOSE(f);
#undef LUISA_FWRITE
#undef LUISA_FCLOSE
    } else {
        LUISA_WARNING("Write file {} failed.", file_path);
    }
}

DefaultBinaryIO::DefaultBinaryIO(Context &&ctx, void *ext) noexcept
    : _ctx(std::move(ctx)),
      _cache_dir{_ctx.create_runtime_subdir(".cache")},
      _data_dir{_ctx.create_runtime_subdir(".data")} {
}

DefaultBinaryIO::~DefaultBinaryIO() noexcept = default;

luisa::unique_ptr<BinaryStream> DefaultBinaryIO::read_shader_bytecode(luisa::string_view name) const noexcept {
    std::filesystem::path local_path{name};
    if (local_path.is_absolute()) {
        return _read(luisa::to_string(name));
    }
    auto file_path = luisa::to_string(_ctx.runtime_directory() / name);
    return _read(file_path);
}

luisa::unique_ptr<BinaryStream> DefaultBinaryIO::read_shader_cache(luisa::string_view name) const noexcept {
    auto file_path = luisa::to_string(_cache_dir / name);
    return _read(file_path);
}

luisa::unique_ptr<BinaryStream> DefaultBinaryIO::read_internal_shader(luisa::string_view name) const noexcept {
    auto file_path = luisa::to_string(_data_dir / name);
    return _read(file_path);
}

luisa::filesystem::path DefaultBinaryIO::write_shader_bytecode(luisa::string_view name, luisa::span<std::byte const> data) const noexcept {
    std::filesystem::path local_path{name};
    if (local_path.is_absolute()) {
        _write(luisa::to_string(name), data);
        return local_path;
    }
    auto file_path = luisa::to_string(_ctx.runtime_directory() / name);
    _write(file_path, data);
    return file_path;
}

luisa::filesystem::path DefaultBinaryIO::write_shader_cache(luisa::string_view name, luisa::span<std::byte const> data) const noexcept {
    auto file_path = luisa::to_string(_cache_dir / name);
    _write(file_path, data);
    return file_path;
}

luisa::filesystem::path DefaultBinaryIO::write_internal_shader(luisa::string_view name, luisa::span<std::byte const> data) const noexcept {
    auto file_path = luisa::to_string(_data_dir / name);
    _write(file_path, data);
    return file_path;
}

}// namespace luisa::compute

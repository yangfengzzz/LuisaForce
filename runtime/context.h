#pragma once

#include "core/stl/memory.h"
#include "core/stl/string.h"
#include "core/stl/hash.h"
#include "core/stl/vector.h"
#include "core/stl/filesystem.h"

namespace luisa {
class BinaryIO;
}// namespace luisa

namespace luisa::compute {

class Device;
struct DeviceConfig;

namespace detail {
class ContextImpl;
}// namespace detail

class LC_RUNTIME_API Context {

private:
    luisa::shared_ptr<detail::ContextImpl> _impl;

public:
    explicit Context(luisa::shared_ptr<luisa::compute::detail::ContextImpl> impl) noexcept;
    // program_path can be first arg from main entry
    explicit Context(luisa::string_view program_path) noexcept;
    explicit Context(const char *program_path) noexcept
        : Context{luisa::string_view{program_path}} {}
    ~Context() noexcept;
    Context(Context &&) noexcept = default;
    Context(const Context &) noexcept = default;
    Context &operator=(Context &&) noexcept = default;
    Context &operator=(const Context &) noexcept = default;
    [[nodiscard]] const auto &impl() const & noexcept { return _impl; }
    [[nodiscard]] auto impl() && noexcept { return std::move(_impl); }
    // runtime directory
    [[nodiscard]] const luisa::filesystem::path &runtime_directory() const noexcept;
    // create subdirectories under the runtime directory
    [[nodiscard]] const luisa::filesystem::path &create_runtime_subdir(luisa::string_view folder_name) const noexcept;
    // Create a virtual device
    [[nodiscard]] Device create_device(const DeviceConfig *settings = nullptr) noexcept;
};

}// namespace luisa::compute

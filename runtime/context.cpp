//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/logging.h"
#include "runtime/context.h"
#include "runtime/device.h"
#include "core/binary_io.h"
#include "core/stl/unordered_map.h"

#if defined(LUISA_PLATFORM_CUDA)
#include "cuda/cuda_device.h"
#endif

#if defined(LUISA_PLATFORM_APPLE)
#include "metal/metal_device.h"
#endif

namespace luisa::compute {
// Make context global, so dynamic modules cannot be redundantly loaded
namespace detail {

class ContextImpl {

public:
    std::filesystem::path runtime_directory;
    luisa::unordered_map<luisa::string, luisa::unique_ptr<std::filesystem::path>> runtime_subdir_paths;
    std::mutex runtime_subdir_mutex;
    std::mutex module_mutex;

    explicit ContextImpl(luisa::string_view program_path) noexcept {
        std::filesystem::path program{program_path};
        {
            auto cp = std::filesystem::canonical(program);
            if (std::filesystem::is_directory(cp)) {
                runtime_directory = std::move(cp);
            } else {
                runtime_directory = std::filesystem::canonical(cp.parent_path());
            }
        }
        LUISA_INFO("Created context for program '{}'.", to_string(program.filename()));
        LUISA_INFO("Runtime directory: {}.", to_string(runtime_directory));
    }
    ~ContextImpl() noexcept = default;
};

}// namespace detail

Context::Context(string_view program_path) noexcept
    : _impl{luisa::make_shared<detail::ContextImpl>(program_path)} {}

Device Context::create_device(const DeviceConfig *settings) noexcept {
    auto interface = create(Context{_impl}, settings);
    auto handle = Device::Handle{
        interface,
        [impl = _impl](auto p) noexcept {
            destroy(p);
        }};
    return Device{std::move(handle)};
}

Context::Context(luisa::shared_ptr<detail::ContextImpl> impl) noexcept
    : _impl{std::move(impl)} {}

Context::~Context() noexcept = default;

const luisa::filesystem::path &Context::runtime_directory() const noexcept {
    return _impl->runtime_directory;
}

const luisa::filesystem::path &Context::create_runtime_subdir(luisa::string_view folder_name) const noexcept {
    std::lock_guard lock{_impl->runtime_subdir_mutex};
    auto iter = _impl->runtime_subdir_paths.try_emplace(
        folder_name,
        luisa::lazy_construct([&]() {
            auto dir = runtime_directory() / folder_name;
            std::error_code ec;
            luisa::filesystem::create_directories(dir, ec);
            if (ec) {
                LUISA_WARNING_WITH_LOCATION(
                    "Failed to create runtime sub-directory '{}': {}.",
                    to_string(dir), ec.message());
            }
            return luisa::make_unique<std::filesystem::path>(std::move(dir));
        }));
    return *iter.first->second;
}

}// namespace luisa::compute

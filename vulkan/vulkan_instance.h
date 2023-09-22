//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <cstdlib>
#include <cstring>

#include <vulkan/vulkan_core.h>
#include "core/stl/memory.h"
#include "core/stl/string.h"
#include "core/stl/vector.h"
#include "core/magic_enum.h"

#define LUISA_CHECK_VULKAN(x)                                            \
    do {                                                                 \
        auto ret = x;                                                    \
        if (ret != VK_SUCCESS) [[unlikely]] {                            \
            if (ret > 0 || ret == VK_ERROR_OUT_OF_DATE_KHR) [[likely]] { \
                LUISA_WARNING_WITH_LOCATION(                             \
                    "Vulkan call `" #x "` returned {}.",                 \
                    ::luisa::to_string(ret));                            \
            } else [[unlikely]] {                                        \
                LUISA_ERROR_WITH_LOCATION(                               \
                    "Vulkan call `" #x "` failed: {}.",                  \
                    ::luisa::to_string(ret));                            \
            }                                                            \
        }                                                                \
    } while (false)

namespace luisa::compute {

static constexpr auto LUISA_REQUIRED_VULKAN_VERSION = VK_API_VERSION_1_2;

struct VulkanDeviceUUID {
    uint8_t bytes[VK_UUID_SIZE];
    [[nodiscard]] auto operator==(const VulkanDeviceUUID &rhs) const noexcept {
        return memcmp(bytes, rhs.bytes, sizeof(bytes)) == 0;
    }
};

static_assert(sizeof(VulkanDeviceUUID) == 16u);

class LC_BACKEND_API VulkanInstance {

private:
    VkInstance _instance{nullptr};
    VkDebugUtilsMessengerEXT _debug_messenger{nullptr};

private:
    VulkanInstance() noexcept;

public:
    // disable copy and move
    VulkanInstance(const VulkanInstance &) noexcept = delete;
    VulkanInstance(VulkanInstance &&) noexcept = delete;
    VulkanInstance &operator=(const VulkanInstance &) noexcept = delete;
    VulkanInstance &operator=(VulkanInstance &&) noexcept = delete;
    ~VulkanInstance() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _instance; }
    [[nodiscard]] auto has_debug_layer() const noexcept { return _debug_messenger != nullptr; }

public:
    [[nodiscard]] static luisa::shared_ptr<VulkanInstance> retain() noexcept;
};

}// namespace luisa::compute

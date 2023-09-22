//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <cuda.h>
#include <atomic>
#include "core/stl/vector.h"

#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN

#include "vulkan_instance.h"

namespace luisa::compute::cuda {

class CUDASemaphoreManager;

class CUDASemaphore {

    friend class CUDASemaphoreManager;

private:
    VkDevice _device;
    VkSemaphore _vk_semaphore;
    CUexternalSemaphore _cuda_semaphore;

public:
    CUDASemaphore(VkDevice device,
              VkSemaphore vk_semaphore,
              CUexternalSemaphore cuda_semaphore) noexcept;
    [[nodiscard]] auto handle() const noexcept { return _cuda_semaphore; }
    void notify(uint64_t value) noexcept;
    void signal(CUstream stream, uint64_t value) noexcept;
    void wait(CUstream stream, uint64_t value) noexcept;
    void synchronize(uint64_t value) noexcept;
    [[nodiscard]] uint64_t signaled_value() noexcept;
    [[nodiscard]] bool is_completed(uint64_t value) noexcept;
};

class CUDASemaphoreManager {

private:
    luisa::shared_ptr<VulkanInstance> _instance;
    VkPhysicalDevice _physical_device{nullptr};
    VkDevice _device{nullptr};
    uint64_t _addr_vkGetSemaphoreHandle{0u};
    std::atomic<size_t> _count{0u};

public:
    explicit CUDASemaphoreManager(const CUuuid &uuid) noexcept;
    ~CUDASemaphoreManager() noexcept;
    CUDASemaphoreManager(CUDASemaphoreManager &&) noexcept = delete;
    CUDASemaphoreManager(const CUDASemaphoreManager &) noexcept = delete;
    CUDASemaphoreManager &operator=(CUDASemaphoreManager &&) noexcept = delete;
    CUDASemaphoreManager &operator=(const CUDASemaphoreManager &) noexcept = delete;
    [[nodiscard]] CUDASemaphore *create() noexcept;
    void destroy(CUDASemaphore *event) noexcept;
};

}// namespace luisa::compute::cuda

#else

#error You cannot use CUDA backend without Vulkan. 😢😢😢. For Windows users, get Vulkan SDK from https://www.lunarg.com/vulkan-sdk/

#endif

//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "cuda_context.h"
#include "cuda_util.h"
#include "cuda_builtin/math/cuda_fabric.h"

#include <map>
#include <vector>

namespace luisa::compute::cuda {
struct DeviceInfo {
    static constexpr int kNameLen = 128;

    CUdevice device = -1;
    int ordinal = -1;
    char name[kNameLen] = "";
    int arch = 0;
    int is_uva = 0;
    int is_memory_pool_supported = 0;
};

struct ContextInfo {
    DeviceInfo *device_info = nullptr;

    CUstream stream = nullptr;// created when needed
};

// cached info for all devices, indexed by ordinal
static std::vector<DeviceInfo> g_devices;

// maps CUdevice to DeviceInfo
static std::map<CUdevice, DeviceInfo *> g_device_map;

// cached info for all known contexts
static std::map<CUcontext, ContextInfo> g_contexts;

static inline CUcontext get_current_context() {
    CUcontext ctx;
    if (check_cu(cuCtxGetCurrent(&ctx)))
        return ctx;
    else
        return nullptr;
}

static ContextInfo *get_context_info(CUcontext ctx) {
    if (!ctx) {
        ctx = get_current_context();
        if (!ctx)
            return nullptr;
    }

    auto it = g_contexts.find(ctx);
    if (it != g_contexts.end()) {
        return &it->second;
    } else {
        // previously unseen context, add the info
        ContextGuard guard(ctx, true);
        ContextInfo context_info;
        CUdevice device;
        if (check_cu(cuCtxGetDevice(&device))) {
            context_info.device_info = g_device_map[device];
            auto result = g_contexts.insert(std::make_pair(ctx, context_info));
            return &result.first->second;
        }
    }

    return nullptr;
}

__global__ void memset_kernel(int *dest, int value, size_t n) {
    const size_t tid = wp::grid_index();

    if (tid < n) {
        dest[tid] = value;
    }
}

void memset_device(void *context, void *dest, int value, size_t n, CUstream stream) {
    ContextGuard guard(context);

    if ((n % 4) > 0) {
        // for unaligned lengths fallback to CUDA memset
        check_cuda(cudaMemsetAsync(dest, value, n, stream));
    } else {
        // custom kernel to support 4-byte values (and slightly lower host overhead)
        const size_t num_words = n / 4;
        wp_launch_device(WP_CURRENT_CONTEXT, memset_kernel, stream, num_words, ((int *)dest, value, num_words));
    }
}

void *alloc_device(void *context, size_t s) {
    ContextGuard guard(context);

    void *ptr;
    check_cuda(cudaMalloc(&ptr, s));
    return ptr;
}

void *alloc_temp_device(void *context, size_t s, CUstream stream) {
    // "cudaMallocAsync ignores the current device/context when determining where the allocation will reside. Instead,
    // cudaMallocAsync determines the resident device based on the specified memory pool or the supplied stream."
    ContextGuard guard(context);

    void *ptr;

    if (cuda_context_is_memory_pool_supported(context)) {
        check_cuda(cudaMallocAsync(&ptr, s, stream));
    } else {
        check_cuda(cudaMalloc(&ptr, s));
    }

    return ptr;
}

void free_device(void *context, void *ptr) {
    ContextGuard guard(context);

    check_cuda(cudaFree(ptr));
}

void free_temp_device(void *context, void *ptr, CUstream stream) {
    ContextGuard guard(context);

    if (cuda_context_is_memory_pool_supported(context)) {
        check_cuda(cudaFreeAsync(ptr, stream));
    } else {
        check_cuda(cudaFree(ptr));
    }
}

void memcpy_h2d(void *context, void *dest, void *src, size_t n, CUstream stream) {
    ContextGuard guard(context);

    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyHostToDevice, stream));
}

void memcpy_d2h(void *context, void *dest, void *src, size_t n, CUstream stream) {
    ContextGuard guard(context);

    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToHost, stream));
}

void memcpy_d2d(void *context, void *dest, void *src, size_t n, CUstream stream) {
    ContextGuard guard(context);

    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToDevice, stream));
}

int cuda_device_is_memory_pool_supported(int ordinal) {
    if (ordinal >= 0 && ordinal < int(g_devices.size()))
        return g_devices[ordinal].is_memory_pool_supported;
    return false;
}

void *cuda_context_get_current() {
    return get_current_context();
}

void cuda_context_synchronize(void *context) {
    ContextGuard guard(context);

    check_cu(cuCtxSynchronize());
}

int cuda_context_get_device_ordinal(void *context) {
    ContextInfo *info = get_context_info(static_cast<CUcontext>(context));
    return info && info->device_info ? info->device_info->ordinal : -1;
}

int cuda_context_is_memory_pool_supported(void *context) {
    int ordinal = cuda_context_get_device_ordinal(context);
    if (ordinal != -1) {
        return cuda_device_is_memory_pool_supported(ordinal);
    }
    return 0;
}

}// namespace luisa::compute::cuda
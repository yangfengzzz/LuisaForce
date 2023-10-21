//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <nvrtc.h>
#include <cuda.h>

#include "core/stl/lru_cache.h"
#include "ast/function.h"
#include "cuda_shader_metadata.h"

namespace luisa::compute::cuda {

class CUDADevice;

/**
 * @brief Kernel compiler of CUDA
 * 
 */
class CUDACompiler {

public:
    using Cache = LRUCache<uint64_t /* hash */,
                           luisa::string /* compiled ptx */>;
    static constexpr auto max_cache_item_count = 64u;

private:
    const CUDADevice *_device;
    uint _nvrtc_version;
    mutable luisa::unique_ptr<Cache> _cache;

public:
    explicit CUDACompiler(const CUDADevice *device) noexcept;
    CUDACompiler(CUDACompiler &&) noexcept = default;
    CUDACompiler(const CUDACompiler &) noexcept = delete;
    CUDACompiler &operator=(CUDACompiler &&) noexcept = default;
    CUDACompiler &operator=(const CUDACompiler &) noexcept = delete;
    [[nodiscard]] auto nvrtc_version() const noexcept { return _nvrtc_version; }
    [[nodiscard]] luisa::string compile(const luisa::string &src, const luisa::string &src_filename,
                                        luisa::span<const char *const> options,
                                        const CUDAShaderMetadata *metadata = nullptr) const noexcept;
    [[nodiscard]] uint64_t compute_hash(const luisa::string &src,
                                        luisa::span<const char *const> options) const noexcept;
    [[nodiscard]] static size_t type_size(const Type *type) noexcept;
    [[nodiscard]] auto device() const noexcept { return _device; }
};

}// namespace luisa::compute::cuda

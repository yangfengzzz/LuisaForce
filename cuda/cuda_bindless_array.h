//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <cuda.h>

#include "core/spin_mutex.h"
#include "core/stl.h"
#include "runtime/rhi/sampler.h"
#include "runtime/rhi/command.h"
#include "resource_tracker.h"
#include "cuda_error.h"
#include "cuda_texture.h"

namespace luisa::compute::cuda {

class CUDADevice;
class CUDAStream;
class CUDACommandEncoder;

/**
 * @brief Bindless array of CUDA
 * 
 */
class CUDABindlessArray {

public:
    struct Slot {
        uint64_t buffer;
        size_t size;
        uint64_t tex2d;
        uint64_t tex3d;
    };

    using Binding = CUdeviceptr;

private:
    CUdeviceptr _handle{};
    luisa::vector<CUtexObject> _tex2d_slots;
    luisa::vector<CUtexObject> _tex3d_slots;
    ResourceTracker _texture_tracker;
    luisa::string _name;
    spin_mutex _mutex;

public:
    explicit CUDABindlessArray(size_t capacity) noexcept;
    ~CUDABindlessArray() noexcept;
    [[nodiscard]] auto handle() const noexcept { return _handle; }
    void update(CUDACommandEncoder &encoder, BindlessArrayUpdateCommand *cmd) noexcept;
    [[nodiscard]] auto binding() const noexcept { return _handle; }
    void set_name(luisa::string &&name) noexcept;
};

}// namespace luisa::compute::cuda

//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <fstream>

#include "core/clock.h"
#include "core/binary_io.h"
#include "cuda_error.h"
#include "cuda_device.h"
#include "cuda_compiler.h"

namespace luisa::compute::cuda {

luisa::string CUDACompiler::compile(const luisa::string &src, const luisa::string &src_filename,
                                    luisa::span<const char *const> options,
                                    const CUDAShaderMetadata *metadata) const noexcept {

    Clock clk;

#ifndef NDEBUG
    // in debug mode, we always recompute the hash, so
    // that we can check the hash if metadata is provided
    auto hash = compute_hash(src, options);
    if (metadata) { LUISA_ASSERT(metadata->checksum == hash, "Hash mismatch!"); }
#else
    auto hash = metadata ? metadata->checksum : compute_hash(src, options);
#endif

    if (auto ptx = _cache->fetch(hash)) { return *ptx; }

    nvrtcProgram prog;
    auto filename = src_filename.empty() ? "my_kernel.cu" : src_filename.c_str();
    LUISA_CHECK_NVRTC(nvrtcCreateProgram(
        &prog, src.data(), filename, 0, nullptr, nullptr));
    auto error = nvrtcCompileProgram(prog, static_cast<int>(options.size()), options.data());
    size_t log_size;
    LUISA_CHECK_NVRTC(nvrtcGetProgramLogSize(prog, &log_size));
    if (log_size > 1u) {
        luisa::string log;
        log.resize(log_size - 1);
        LUISA_CHECK_NVRTC(nvrtcGetProgramLog(prog, log.data()));
        LUISA_WARNING_WITH_LOCATION("Compile log:\n{}", log);
    }
    LUISA_CHECK_NVRTC(error);
    size_t ptx_size;
    luisa::string ptx;

    LUISA_CHECK_NVRTC(nvrtcGetPTXSize(prog, &ptx_size));
    ptx.resize(ptx_size - 1u);
    LUISA_CHECK_NVRTC(nvrtcGetPTX(prog, ptx.data()));

    LUISA_CHECK_NVRTC(nvrtcDestroyProgram(&prog));
    LUISA_VERBOSE("CUDACompiler::compile() took {} ms.", clk.toc());
    return ptx;
}

size_t CUDACompiler::type_size(const Type *type) noexcept {
    if (!type->is_custom()) { return type->size(); }
    // TODO: support custom types
    if (type->description() == "LC_IndirectKernelDispatch") {
        LUISA_ERROR_WITH_LOCATION("Not implemented.");
    }
    LUISA_ERROR_WITH_LOCATION("Not implemented.");
}

CUDACompiler::CUDACompiler(const CUDADevice *device) noexcept
    : _device{device},
      _nvrtc_version{[] {
          auto ver_major = 0;
          auto ver_minor = 0;
          LUISA_CHECK_NVRTC(nvrtcVersion(&ver_major, &ver_minor));
          return static_cast<uint>(ver_major * 10000 + ver_minor * 100);
      }()},
      _cache{Cache::create(max_cache_item_count)} {
    LUISA_VERBOSE("CUDA NVRTC version: {}.", _nvrtc_version);
}

uint64_t CUDACompiler::compute_hash(const string &src, luisa::span<const char *const> options) const noexcept {
    auto hash = hash_value(src);
    for (auto o : options) { hash = hash_value(o, hash); }
    return hash;
}

}// namespace luisa::compute::cuda

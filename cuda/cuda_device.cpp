//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <fstream>
#include <future>

#include "core/clock.h"
#include "core/binary_io.h"
#include "runtime/rhi/sampler.h"
#include "runtime/bindless_array.h"
#include "runtime/dispatch_buffer.h"

#include "string_scratch.h"
#include "cuda_error.h"
#include "cuda_device.h"
#include "cuda_buffer.h"
#include "cuda_stream.h"
#include "cuda_semaphore.h"
#include "cuda_codegen_ast.h"
#include "cuda_compiler.h"
#include "cuda_bindless_array.h"
#include "cuda_command_encoder.h"
#include "cuda_texture.h"
#include "cuda_shader_native.h"
#include "cuda_shader_metadata.h"
#include "cuda_swapchain.h"

#include "cuda_builtin_kernels/hashgrid.h"

#define LUISA_CUDA_KERNEL_DEBUG 1

#ifndef NDEBUG
#define LUISA_CUDA_DUMP_SOURCE 1
#else
static const bool LUISA_CUDA_DUMP_SOURCE = ([] {
    // read env LUISA_DUMP_SOURCE
    auto env = std::getenv("LUISA_DUMP_SOURCE");
    if (env == nullptr) return false;
    return std::string_view{env} == "1";
})();
#endif

namespace luisa::compute::cuda {

[[nodiscard]] static auto cuda_array_format(PixelFormat format) noexcept {
    switch (format) {
        case PixelFormat::R8SInt: return CU_AD_FORMAT_SIGNED_INT8;
        case PixelFormat::R8UInt: [[fallthrough]];
        case PixelFormat::R8UNorm: return CU_AD_FORMAT_UNSIGNED_INT8;
        case PixelFormat::RG8SInt: return CU_AD_FORMAT_SIGNED_INT8;
        case PixelFormat::RG8UInt: [[fallthrough]];
        case PixelFormat::RG8UNorm: return CU_AD_FORMAT_UNSIGNED_INT8;
        case PixelFormat::RGBA8SInt: return CU_AD_FORMAT_SIGNED_INT8;
        case PixelFormat::RGBA8UInt: [[fallthrough]];
        case PixelFormat::RGBA8UNorm: return CU_AD_FORMAT_UNSIGNED_INT8;
        case PixelFormat::R16SInt: return CU_AD_FORMAT_SIGNED_INT16;
        case PixelFormat::R16UInt: [[fallthrough]];
        case PixelFormat::R16UNorm: return CU_AD_FORMAT_UNSIGNED_INT16;
        case PixelFormat::RG16SInt: return CU_AD_FORMAT_SIGNED_INT16;
        case PixelFormat::RG16UInt: [[fallthrough]];
        case PixelFormat::RG16UNorm: return CU_AD_FORMAT_UNSIGNED_INT16;
        case PixelFormat::RGBA16SInt: return CU_AD_FORMAT_SIGNED_INT16;
        case PixelFormat::RGBA16UInt: [[fallthrough]];
        case PixelFormat::RGBA16UNorm: return CU_AD_FORMAT_UNSIGNED_INT16;
        case PixelFormat::R32SInt: return CU_AD_FORMAT_SIGNED_INT32;
        case PixelFormat::R32UInt: return CU_AD_FORMAT_UNSIGNED_INT32;
        case PixelFormat::RG32SInt: return CU_AD_FORMAT_SIGNED_INT32;
        case PixelFormat::RG32UInt: return CU_AD_FORMAT_UNSIGNED_INT32;
        case PixelFormat::RGBA32SInt: return CU_AD_FORMAT_SIGNED_INT32;
        case PixelFormat::RGBA32UInt: return CU_AD_FORMAT_UNSIGNED_INT32;
        case PixelFormat::R16F: return CU_AD_FORMAT_HALF;
        case PixelFormat::RG16F: return CU_AD_FORMAT_HALF;
        case PixelFormat::RGBA16F: return CU_AD_FORMAT_HALF;
        case PixelFormat::R32F: return CU_AD_FORMAT_FLOAT;
        case PixelFormat::RG32F: return CU_AD_FORMAT_FLOAT;
        case PixelFormat::RGBA32F: return CU_AD_FORMAT_FLOAT;
        case PixelFormat::BC4UNorm: return CU_AD_FORMAT_BC4_UNORM;
        case PixelFormat::BC5UNorm: return CU_AD_FORMAT_BC5_UNORM;
        case PixelFormat::BC6HUF16: return CU_AD_FORMAT_BC6H_UF16;
        case PixelFormat::BC7UNorm: return CU_AD_FORMAT_BC7_UNORM;
        default: break;
    }
    LUISA_ERROR_WITH_LOCATION("Invalid pixel format 0x{:02x}.",
                              luisa::to_underlying(format));
}

CUDADevice::CUDADevice(Context &&ctx,
                       size_t device_id,
                       const BinaryIO *io) noexcept
    : DeviceInterface{std::move(ctx)}, _handle{device_id}, _io{io} {
    // provide a default binary IO
    if (_io == nullptr) {
        _default_io = luisa::make_unique<DefaultBinaryIO>(context());
        _io = _default_io.get();
    }
    _compiler = luisa::make_unique<CUDACompiler>(this);

    // load cudadevrt
#ifdef LUISA_PLATFORM_WINDOWS
    auto device_runtime_lib_path = context().runtime_directory() / "cudadevrt.lib";
#else
    auto device_runtime_lib_path = context().runtime_directory() / "libcudadevrt.a";
#endif
    std::ifstream devrt_file{device_runtime_lib_path, std::ios::binary};
    if (!devrt_file.is_open()) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to load CUDA device runtime library '{}'. "
            "Indirect kernel dispatch will not be available.",
            device_runtime_lib_path.string());
    } else {
        _cudadevrt_library = luisa::string{std::istreambuf_iterator<char>{devrt_file},
                                           std::istreambuf_iterator<char>{}};
    }

    // event pool
    _semaphore_manager = luisa::make_unique<CUDASemaphoreManager>(handle().uuid());
}

CUDADevice::~CUDADevice() noexcept {
    with_handle([this] {
        LUISA_CHECK_CUDA(cuCtxSynchronize());
    });
}

BufferCreationInfo CUDADevice::create_buffer(const Type *element, size_t elem_count) noexcept {
    BufferCreationInfo info{};
    elem_count = std::max<size_t>(elem_count, 1u);
    if (element == Type::of<IndirectKernelDispatch>()) {
        auto buffer = with_handle([elem_count] {
            return new_with_allocator<CUDAIndirectDispatchBuffer>(elem_count);
        });
        info.handle = reinterpret_cast<uint64_t>(buffer);
        info.native_handle = reinterpret_cast<void *>(buffer->handle());
        info.element_stride = sizeof(CUDAIndirectDispatchBuffer::Dispatch);
        info.total_size_bytes = buffer->size_bytes();
    } else if (element == Type::of<void>()) {
        info.element_stride = 1;
        info.total_size_bytes = elem_count;
        auto buffer = with_handle([size = info.total_size_bytes] {
            return new_with_allocator<CUDABuffer>(size);
        });
        info.handle = reinterpret_cast<uint64_t>(buffer);
        info.native_handle = reinterpret_cast<void *>(buffer->handle());
    } else {
        info.element_stride = CUDACompiler::type_size(element);
        info.total_size_bytes = info.element_stride * elem_count;
        auto buffer = with_handle([size = info.total_size_bytes] {
            return new_with_allocator<CUDABuffer>(size);
        });
        info.handle = reinterpret_cast<uint64_t>(buffer);
        info.native_handle = reinterpret_cast<void *>(buffer->handle());
    }
    return info;
}

void CUDADevice::destroy_buffer(uint64_t handle) noexcept {
    with_handle([buffer = reinterpret_cast<CUDABufferBase *>(handle)] {
        delete_with_allocator(buffer);
    });
}

ResourceCreationInfo CUDADevice::create_texture(PixelFormat format, uint dimension, uint width, uint height, uint depth, uint mipmap_levels, bool simultaneous_access) noexcept {
    auto p = with_handle([=] {
        auto array_format = cuda_array_format(format);
        auto channels = pixel_format_channel_count(format);
        CUDA_ARRAY3D_DESCRIPTOR array_desc{};
        array_desc.Width = width;
        array_desc.Height = height;
        array_desc.Depth = dimension == 2u ? 0u : depth;
        array_desc.Format = array_format;
        array_desc.NumChannels = channels;
        if (!is_block_compressed(format)) {
            array_desc.Flags = CUDA_ARRAY3D_SURFACE_LDST;
        }
        auto array_handle = [&] {
            if (mipmap_levels == 1u) {
                CUarray handle{nullptr};
                LUISA_CHECK_CUDA(cuArray3DCreate(&handle, &array_desc));
                return reinterpret_cast<uint64_t>(handle);
            }
            CUmipmappedArray handle{nullptr};
            LUISA_CHECK_CUDA(cuMipmappedArrayCreate(&handle, &array_desc, mipmap_levels));
            return reinterpret_cast<uint64_t>(handle);
        }();
        return new_with_allocator<CUDATexture>(
            array_handle, make_uint3(width, height, depth),
            format, mipmap_levels);
    });
    return {.handle = reinterpret_cast<uint64_t>(p),
            .native_handle = reinterpret_cast<void *>(p->handle())};
}

void CUDADevice::destroy_texture(uint64_t handle) noexcept {
    with_handle([array = reinterpret_cast<CUDATexture *>(handle)] {
        delete_with_allocator(array);
    });
}

ResourceCreationInfo CUDADevice::create_bindless_array(size_t size) noexcept {
    auto p = with_handle([size] { return new_with_allocator<CUDABindlessArray>(size); });
    return {.handle = reinterpret_cast<uint64_t>(p),
            .native_handle = reinterpret_cast<void *>(p->handle())};
}

void CUDADevice::destroy_bindless_array(uint64_t handle) noexcept {
    with_handle([array = reinterpret_cast<CUDABindlessArray *>(handle)] {
        delete_with_allocator(array);
    });
}

ResourceCreationInfo CUDADevice::create_stream(StreamTag stream_tag) noexcept {
#ifndef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
    if (stream_tag == StreamTag::GRAPHICS) {
        LUISA_WARNING_WITH_LOCATION("Swapchains are not enabled on CUDA backend, "
                                    "Graphics streams might not work properly.");
    }
#endif
    auto p = with_handle([&] { return new_with_allocator<CUDAStream>(this); });
    return {.handle = reinterpret_cast<uint64_t>(p),
            .native_handle = p->handle()};
}

void CUDADevice::destroy_stream(uint64_t handle) noexcept {
    with_handle([stream = reinterpret_cast<CUDAStream *>(handle)] {
        delete_with_allocator(stream);
    });
}

void CUDADevice::synchronize_stream(uint64_t stream_handle) noexcept {
    with_handle([stream = reinterpret_cast<CUDAStream *>(stream_handle)] {
        stream->synchronize();
    });
}

void CUDADevice::dispatch(uint64_t stream_handle, CommandList &&list) noexcept {
    if (!list.empty()) {
        with_handle([stream = reinterpret_cast<CUDAStream *>(stream_handle),
                     list = std::move(list)]() mutable noexcept {
            stream->dispatch(std::move(list));
        });
    }
}

SwapchainCreationInfo CUDADevice::create_swapchain(uint64_t window_handle, uint64_t stream_handle,
                                                   uint width, uint height,
                                                   bool allow_hdr, bool vsync,
                                                   uint back_buffer_size) noexcept {
#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
    auto chain = with_handle([&] {
        return new_with_allocator<CUDASwapchain>(
            this, window_handle, width, height,
            allow_hdr, vsync, back_buffer_size);
    });
    SwapchainCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(chain);
    info.native_handle = chain;
    info.storage = chain->pixel_storage();
    return info;
#else
    LUISA_ERROR_WITH_LOCATION("Swapchains are not enabled on the CUDA backend. "
                              "You need to enable the GUI module and install "
                              "the Vulkan SDK (>= 1.1) to enable it.");
#endif
}

void CUDADevice::destroy_swap_chain(uint64_t handle) noexcept {
#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
    with_handle([chain = reinterpret_cast<CUDASwapchain *>(handle)] {
        delete_with_allocator(chain);
    });
#else
    LUISA_ERROR_WITH_LOCATION("Swapchains are not enabled on the CUDA backend. "
                              "You need to enable the GUI module and install "
                              "the Vulkan SDK (>= 1.1) to enable it.");
#endif
}

void CUDADevice::present_display_in_stream(uint64_t stream_handle, uint64_t swapchain_handle, uint64_t image_handle) noexcept {
#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
    with_handle([stream = reinterpret_cast<CUDAStream *>(stream_handle),
                 chain = reinterpret_cast<CUDASwapchain *>(swapchain_handle),
                 image = reinterpret_cast<CUDATexture *>(image_handle)] {
        chain->present(stream, image);
    });
#else
    LUISA_ERROR_WITH_LOCATION("Swapchains are not enabled on the CUDA backend. "
                              "You need to enable the GUI module and install "
                              "the Vulkan SDK (>= 1.1) to enable it.");
#endif
}

[[nodiscard]] inline luisa::optional<CUDAShaderMetadata>
parse_shader_metadata(luisa::string_view data,
                      luisa::string_view name) noexcept {
    if (data.empty()) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to parse shader metadata for '{}': "
            "PTX source is empty.",
            name);
        return luisa::nullopt;
    }
    constexpr luisa::string_view metadata_prefix = "// METADATA: ";
    if (!data.starts_with(metadata_prefix)) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to parse shader metadata for '{}': "
            "PTX source does not start with metadata prefix.",
            name);
        return luisa::nullopt;
    }
    auto m = data.substr(metadata_prefix.size());
    auto metadata = deserialize_cuda_shader_metadata(m);
    if (!metadata) {
        LUISA_WARNING_WITH_LOCATION(
            "Failed to parse shader metadata for '{}': "
            "invalid metadata string '{}'.",
            name, m);
        return luisa::nullopt;
    }
    return metadata;
}

template<bool allow_update_expected_metadata>
[[nodiscard]] inline luisa::string load_shader_ptx(BinaryStream *metadata_stream,
                                                   BinaryStream *ptx_stream,
                                                   luisa::string_view name,
                                                   bool warn_not_found,
                                                   std::conditional_t<allow_update_expected_metadata,
                                                                      CUDAShaderMetadata,
                                                                      const CUDAShaderMetadata> &
                                                       expected_metadata) noexcept {
    // check if the stream is valid
    if (ptx_stream == nullptr || ptx_stream->length() == 0u ||
        metadata_stream == nullptr || metadata_stream->length() == 0u) {
        if (warn_not_found) {
            LUISA_WARNING_WITH_LOCATION(
                "Shader '{}' is not found in cache. "
                "This may be caused by a mismatch between the shader source and the cached binary. "
                "The shader will be recompiled.",
                name);
        } else {
            LUISA_VERBOSE("Shader '{}' is not found in cache. "
                          "The shader will be recompiled.",
                          name);
        }
        return {};
    }
    // read metadata string from stream
    luisa::string meta_data;
    meta_data.resize(metadata_stream->length());
    metadata_stream->read(luisa::span{
        reinterpret_cast<std::byte *>(meta_data.data()),
        meta_data.size() * sizeof(char)});

    // read ptx
    luisa::string ptx_data;
    ptx_data.resize(ptx_stream->length());
    ptx_stream->read(luisa::span{
        reinterpret_cast<std::byte *>(ptx_data.data()),
        ptx_data.size() * sizeof(char)});

    // parse metadata
    auto metadata = parse_shader_metadata(meta_data, name);
    if (!metadata) {
        LUISA_WARNING_WITH_LOCATION(
            "Shader '{}' is found in cache, but its metadata is invalid. "
            "This may be caused by a mismatch between the shader source and the cached binary. "
            "The shader will be recompiled.",
            name);
        return {};
    }
    // update the empty fields in metadata
    if constexpr (allow_update_expected_metadata) {
        if (expected_metadata.checksum == 0u) { expected_metadata.checksum = metadata->checksum; }
        if (expected_metadata.kind == CUDAShaderMetadata::Kind::UNKNOWN) { expected_metadata.kind = metadata->kind; }
        expected_metadata.enable_debug = metadata->enable_debug;
        if (all(expected_metadata.block_size == 0u)) { expected_metadata.block_size = metadata->block_size; }
        if (expected_metadata.argument_types.empty()) { expected_metadata.argument_types = metadata->argument_types; }
        if (expected_metadata.argument_usages.empty()) { expected_metadata.argument_usages = metadata->argument_usages; }
    }
    // examine the metadata
    if (*metadata != expected_metadata) {
        LUISA_WARNING_WITH_LOCATION(
            "Shader '{}' is found in cache, but its metadata '{}' do not match the expected '{}'. "
            "This may be caused by a mismatch between the shader source and the cached binary. "
            "The shader will be recompiled.",
            name, serialize_cuda_shader_metadata(*metadata),
            serialize_cuda_shader_metadata(expected_metadata));
        return {};
    }
    // return the ptx string
    return ptx_data;
}

ShaderCreationInfo CUDADevice::_create_shader(std::string name,
                                              const string &source, const ShaderOption &option,
                                              luisa::span<const char *const> nvrtc_options,
                                              const CUDAShaderMetadata &expected_metadata,
                                              luisa::vector<ShaderDispatchCommand::Argument> bound_arguments) noexcept {

    // generate a default name if not specified
    auto uses_user_path = !name.empty();
    if (!uses_user_path) { name = fmt::format("kernel_{:016x}.ptx",
                                              expected_metadata.checksum); }
    if (!name.ends_with(".ptx") &&
        !name.ends_with(".PTX")) { name.append(".ptx"); }
    auto metadata_name = fmt::format("{}.metadata", name);

    // try disk cache
    auto ptx = [&] {
        luisa::unique_ptr<BinaryStream> ptx_stream;
        luisa::unique_ptr<BinaryStream> metadata_stream;
        if (uses_user_path) {
            ptx_stream = _io->read_shader_bytecode(name);
            metadata_stream = _io->read_shader_bytecode(metadata_name);
        } else if (option.enable_cache) {
            ptx_stream = _io->read_shader_cache(name);
            metadata_stream = _io->read_shader_cache(metadata_name);
        }
        return load_shader_ptx<false>(
            metadata_stream.get(), ptx_stream.get(),
            name, false, expected_metadata);
    }();

    // compile if not found in cache
    if (ptx.empty()) {
        luisa::filesystem::path src_dump_path;
        if (option.enable_debug_info || LUISA_CUDA_DUMP_SOURCE) {
            luisa::span src_data{reinterpret_cast<const std::byte *>(source.data()), source.size()};
            auto src_name = fmt::format("{}.cu", name);
            if (uses_user_path) {
                src_dump_path = _io->write_shader_bytecode(src_name, src_data);
            } else if (option.enable_cache) {
                src_dump_path = _io->write_shader_cache(src_name, src_data);
            }
        }
        luisa::string src_filename{src_dump_path.string()};
        ptx = _compiler->compile(source, src_filename, nvrtc_options, &expected_metadata);
        if (!ptx.empty()) {
            luisa::span ptx_data{reinterpret_cast<const std::byte *>(ptx.data()), ptx.size()};
            auto metadata = fmt::format("// METADATA: {}\n\n", serialize_cuda_shader_metadata(expected_metadata));
            luisa::span metadata_data{reinterpret_cast<const std::byte *>(metadata.data()), metadata.size()};
            if (uses_user_path) {
                static_cast<void>(_io->write_shader_bytecode(name, ptx_data));
                static_cast<void>(_io->write_shader_bytecode(metadata_name, metadata_data));
            } else if (option.enable_cache) {
                static_cast<void>(_io->write_shader_cache(name, ptx_data));
                static_cast<void>(_io->write_shader_cache(metadata_name, metadata_data));
            }
        }
    }

    if (option.compile_only) {// no shader object should be created
        return ShaderCreationInfo::make_invalid();
    }

    // create the shader object
    auto p = with_handle([&]() noexcept -> CUDAShader * {
        return new_with_allocator<CUDAShaderNative>(
            this, ptx.data(), ptx.size(), "kernel_main",
            expected_metadata.block_size,
            expected_metadata.argument_usages,
            std::move(bound_arguments));
    });
#ifndef NDEBUG
    p->set_name(std::move(name));
#endif
    ShaderCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(p);
    info.native_handle = p->handle();
    info.block_size = expected_metadata.block_size;
    return info;
}

ShaderCreationInfo CUDADevice::create_shader(const ShaderOption &option, Function kernel) noexcept {

    // codegen
    Clock clk;
    StringScratch scratch;
    CUDACodegenAST codegen{scratch, !_cudadevrt_library.empty()};
    codegen.emit(kernel, option.native_include);
    LUISA_VERBOSE("Generated CUDA source in {} ms.", clk.toc());

    // process bound arguments
    luisa::vector<ShaderDispatchCommand::Argument> bound_arguments;
    bound_arguments.reserve(kernel.bound_arguments().size());
    for (auto &&arg : kernel.bound_arguments()) {
        luisa::visit(
            [&bound_arguments]<typename T>(T binding) noexcept {
                ShaderDispatchCommand::Argument argument{};
                if constexpr (std::is_same_v<T, Function::BufferBinding>) {
                    argument.tag = ShaderDispatchCommand::Argument::Tag::BUFFER;
                    argument.buffer.handle = binding.handle;
                    argument.buffer.offset = binding.offset;
                    argument.buffer.size = binding.size;
                } else if constexpr (std::is_same_v<T, Function::TextureBinding>) {
                    argument.tag = ShaderDispatchCommand::Argument::Tag::TEXTURE;
                    argument.texture.handle = binding.handle;
                    argument.texture.level = binding.level;
                } else if constexpr (std::is_same_v<T, Function::BindlessArrayBinding>) {
                    argument.tag = ShaderDispatchCommand::Argument::Tag::BINDLESS_ARRAY;
                    argument.bindless_array.handle = binding.handle;
                } else if constexpr (std::is_same_v<T, Function::HashGridBinding>) {
                    argument.tag = ShaderDispatchCommand::Argument::Tag::HASH_GRID;
                    argument.hash_grid.handle = binding.handle;
                } else {
                    LUISA_ERROR_WITH_LOCATION("Unsupported binding type.");
                }
                bound_arguments.emplace_back(argument);
            },
            arg);
    }

    // NVRTC nvrtc_options
    auto sm_option = fmt::format("-arch=compute_{}", _handle.compute_capability());
    auto nvrtc_version_option = fmt::format("-DLC_NVRTC_VERSION={}", _compiler->nvrtc_version());
    luisa::vector<const char *> nvrtc_options {
        sm_option.c_str(),
            nvrtc_version_option.c_str(),
            "--std=c++17",
            "-default-device",
            "-restrict",
            "-extra-device-vectorization",
            "-dw",
            "-w",
            "-ewp",
#if !defined(NDEBUG) && LUISA_CUDA_KERNEL_DEBUG
            "-DLUISA_DEBUG=1",
#endif
    };

    if (option.enable_debug_info) {
        nvrtc_options.emplace_back("-lineinfo");
#if defined(NDEBUG) || !LUISA_CUDA_KERNEL_DEBUG
        nvrtc_options.emplace_back("-DLUISA_DEBUG=1");
#endif
    }
    if (option.enable_fast_math) {
        nvrtc_options.emplace_back("-use_fast_math");
    }

    // include path
    auto include_dir = std::filesystem::current_path().string() + "/../../../../cuda/cuda_builtin";
    auto include_opt = fmt::format("--include-path={}", include_dir);
    nvrtc_options.push_back(include_opt.c_str());

    // compute hash
    auto src_hash = _compiler->compute_hash(scratch.string(), nvrtc_options);

    // create metadata
    CUDAShaderMetadata metadata{
        .checksum = src_hash,
        .kind = CUDAShaderMetadata::Kind::COMPUTE,
        .enable_debug = option.enable_debug_info,
        .block_size = kernel.block_size(),
        .argument_types = [kernel] {
            luisa::vector<luisa::string> types;
            types.reserve(kernel.arguments().size());
            std::transform(kernel.arguments().cbegin(), kernel.arguments().cend(), std::back_inserter(types),
                           [](auto &&arg) noexcept { return luisa::string{arg.type()->description()}; });
            return types; }(),
        .argument_usages = [kernel] {
            luisa::vector<Usage> usages;
            usages.reserve(kernel.arguments().size());
            std::transform(kernel.arguments().cbegin(), kernel.arguments().cend(), std::back_inserter(usages),
                           [kernel](auto &&arg) noexcept { return kernel.variable_usage(arg.uid()); });
            return usages; }(),
    };
    return _create_shader(option.name, scratch.string(),
                          option, nvrtc_options,
                          metadata, std::move(bound_arguments));
}

ShaderCreationInfo CUDADevice::load_shader(luisa::string_view name_in,
                                           luisa::span<const Type *const> arg_types) noexcept {

    std::string name{name_in};
    if (!name.ends_with(".ptx") &&
        !name.ends_with(".PTX")) { name.append(".ptx"); }
    auto metadata_name = fmt::format("{}.metadata", name);

    // prepare (incomplete) metadata
    CUDAShaderMetadata metadata{
        .checksum = 0u,
        .kind = CUDAShaderMetadata::Kind::UNKNOWN,
        .block_size = uint3{1u, 1u, 1u},
        .argument_types = [arg_types] {
            luisa::vector<luisa::string> types;
            types.reserve(arg_types.size());
            std::transform(arg_types.cbegin(), arg_types.cend(), std::back_inserter(types),
                           [](auto &&arg) noexcept { return luisa::string{arg->description()}; });
            return types; }(),
    };

    // load ptx
    auto ptx = [&] {
        auto metadata_stream = _io->read_shader_bytecode(metadata_name);
        auto ptx_stream = _io->read_shader_bytecode(name);
        return load_shader_ptx<true>(metadata_stream.get(), ptx_stream.get(), name, true, metadata);
    }();
    if (ptx.empty()) {
        LUISA_WARNING_WITH_LOCATION("Failed to load shader bytecode from {}.", name);
        return ShaderCreationInfo::make_invalid();
    }

    // check argument count
    if (metadata.argument_types.size() != arg_types.size()) {
        LUISA_WARNING_WITH_LOCATION("Argument count mismatch when loading shader {}.", name);
        return ShaderCreationInfo::make_invalid();
    }

    // create shader
    auto p = with_handle([&]() noexcept -> CUDAShader * {
        return new_with_allocator<CUDAShaderNative>(
            this, ptx.data(), ptx.size(), "kernel_main",
            metadata.block_size, metadata.argument_usages);
    });
#ifndef NDEBUG
    p->set_name(std::move(name));
#endif
    ShaderCreationInfo info{};
    info.handle = reinterpret_cast<uint64_t>(p);
    info.native_handle = p->handle();
    info.block_size = metadata.block_size;
    return info;
}

Usage CUDADevice::shader_argument_usage(uint64_t handle, size_t index) noexcept {
    return reinterpret_cast<const CUDAShader *>(handle)->argument_usage(index);
}

void CUDADevice::destroy_shader(uint64_t handle) noexcept {
    with_handle([shader = reinterpret_cast<CUDAShader *>(handle)] {
        delete_with_allocator(shader);
    });
}

ResourceCreationInfo CUDADevice::create_event() noexcept {
    auto event_handle = with_handle([this] {
        return _semaphore_manager->create();
    });
    return {.handle = reinterpret_cast<uint64_t>(event_handle),
            .native_handle = event_handle->handle()};
}

void CUDADevice::destroy_event(uint64_t handle) noexcept {
    with_handle([this, handle] {
        auto event = reinterpret_cast<CUDASemaphore *>(handle);
        _semaphore_manager->destroy(event);
    });
}

void CUDADevice::signal_event(uint64_t handle, uint64_t stream_handle, uint64_t value) noexcept {
    with_handle([=] {
        auto event = reinterpret_cast<CUDASemaphore *>(handle);
        auto stream = reinterpret_cast<CUDAStream *>(stream_handle);
        stream->signal(event, value);
    });
}

void CUDADevice::wait_event(uint64_t handle, uint64_t stream_handle, uint64_t value) noexcept {
    with_handle([=] {
        auto event = reinterpret_cast<CUDASemaphore *>(handle);
        auto stream = reinterpret_cast<CUDAStream *>(stream_handle);
        stream->wait(event, value);
    });
}

bool CUDADevice::is_event_completed(uint64_t handle, uint64_t value) const noexcept {
    auto event = reinterpret_cast<CUDASemaphore *>(handle);
    return event->is_completed(value);
}

void CUDADevice::synchronize_event(uint64_t handle, uint64_t value) noexcept {
    auto event = reinterpret_cast<CUDASemaphore *>(handle);
    event->synchronize(value);
}

string CUDADevice::query(luisa::string_view property) noexcept {
    if (property == "device_name") {
        return "cuda";
    }
    LUISA_WARNING_WITH_LOCATION("Unknown device property '{}'.", property);
    return {};
}

DeviceExtension *CUDADevice::extension(luisa::string_view name) noexcept {

#define LUISA_COMPUTE_CREATE_CUDA_EXTENSION(ext, v)                         \
    if (name == ext##Ext::name) {                                           \
        std::scoped_lock lock{_ext_mutex};                                  \
        if (v == nullptr) { v = luisa::make_unique<CUDA##ext##Ext>(this); } \
        return v.get();                                                     \
    }
#undef LUISA_COMPUTE_CREATE_CUDA_EXTENSION

    LUISA_WARNING_WITH_LOCATION("Unknown device extension '{}'.", name);
    return nullptr;
}

namespace detail {

static constexpr auto required_cuda_version_major = 11;
static constexpr auto required_cuda_version_minor = 7;
static constexpr auto required_cuda_version = required_cuda_version_major * 1000 + required_cuda_version_minor * 10;

static void initialize() {
    // global init
    static std::once_flag flag;
    std::call_once(flag, [] {
        // CUDA
        LUISA_CHECK_CUDA(cuInit(0));
        // check driver version
        auto driver_version = 0;
        LUISA_CHECK_CUDA(cuDriverGetVersion(&driver_version));
        auto driver_version_major = driver_version / 1000;
        auto driver_version_minor = (driver_version % 1000) / 10;
        LUISA_ASSERT(driver_version >= required_cuda_version,
                     "CUDA driver version {}.{} is too old (>= {}.{} required). "
                     "Please update your driver.",
                     driver_version_major, driver_version_minor,
                     required_cuda_version_major, required_cuda_version_minor);
        LUISA_VERBOSE("Successfully initialized CUDA "
                      "backend with driver version {}.{}.",
                      driver_version_major, driver_version_minor);
    });
}

}// namespace detail

CUDADevice::Handle::Handle(size_t index) noexcept {

    detail::initialize();

    // cuda
    auto driver_version = 0;
    LUISA_CHECK_CUDA(cuDriverGetVersion(&driver_version));
    _driver_version = driver_version;

    auto device_count = 0;
    LUISA_CHECK_CUDA(cuDeviceGetCount(&device_count));
    if (device_count == 0) {
        LUISA_ERROR_WITH_LOCATION("No available device found for CUDA backend.");
    }
    if (index >= device_count) {
        LUISA_WARNING_WITH_LOCATION(
            "Invalid device index {} (device count = {}). Limiting to {}.",
            index, device_count, device_count - 1);
        index = device_count - 1;
    }
    _device_index = index;
    LUISA_CHECK_CUDA(cuDeviceGet(&_device, index));
    auto compute_cap_major = 0;
    auto compute_cap_minor = 0;
    LUISA_CHECK_CUDA(cuDeviceGetUuid(&_uuid, _device));
    LUISA_CHECK_CUDA(cuDeviceGetAttribute(&compute_cap_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, _device));
    LUISA_CHECK_CUDA(cuDeviceGetAttribute(&compute_cap_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, _device));

    auto can_map_host_memory = 0;
    auto unified_addressing = 0;
    LUISA_CHECK_CUDA(cuDeviceGetAttribute(&can_map_host_memory, CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, _device));
    LUISA_CHECK_CUDA(cuDeviceGetAttribute(&unified_addressing, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, _device));
    LUISA_VERBOSE("Device {} can map host memory: {}", index, can_map_host_memory != 0);
    LUISA_VERBOSE("Device {} supports unified addressing: {}", index, unified_addressing != 0);

    auto format_uuid = [](auto uuid) noexcept {
        luisa::string result;
        result.reserve(36u);
        auto count = 0u;
        for (auto c : uuid.bytes) {
            if (count == 4u || count == 6u || count == 8u || count == 10u) {
                result.append("-");
            }
            result.append(fmt::format("{:02x}", static_cast<uint>(c) & 0xffu));
            count++;
        }
        return result;
    };

    LUISA_INFO("Created CUDA device at index {}: {} "
               "(driver = {}, capability = {}.{}, uuid = {}).",
               index, name(), driver_version,
               compute_cap_major, compute_cap_minor,
               format_uuid(_uuid));
    _compute_capability = 10u * compute_cap_major + compute_cap_minor;
    LUISA_CHECK_CUDA(cuDevicePrimaryCtxRetain(&_context, _device));
}

CUDADevice::Handle::~Handle() noexcept {
    LUISA_CHECK_CUDA(cuDevicePrimaryCtxRelease(_device));
    LUISA_VERBOSE("Destroyed CUDA device: {}.", name());
}

std::string_view CUDADevice::Handle::name() const noexcept {
    static constexpr auto device_name_length = 1024u;
    static thread_local char device_name[device_name_length];
    LUISA_CHECK_CUDA(cuDeviceGetName(device_name, device_name_length, _device));
    return device_name;
}

void CUDADevice::set_name(luisa::compute::Resource::Tag resource_tag,
                          uint64_t resource_handle,
                          luisa::string_view name) noexcept {
    with_handle([tag = resource_tag,
                 handle = resource_handle,
                 name = std::string{name}]() mutable noexcept {
        switch (tag) {
            case Resource::Tag::BUFFER:
                reinterpret_cast<CUDABufferBase *>(handle)->set_name(std::move(name));
                break;
            case Resource::Tag::TEXTURE:
                reinterpret_cast<CUDATexture *>(handle)->set_name(std::move(name));
                break;
            case Resource::Tag::BINDLESS_ARRAY:
                reinterpret_cast<CUDABindlessArray *>(handle)->set_name(std::move(name));
                break;
            case Resource::Tag::STREAM:
                reinterpret_cast<CUDAStream *>(handle)->set_name(std::move(name));
                break;
            case Resource::Tag::EVENT:
                break;
            case Resource::Tag::SHADER:
                reinterpret_cast<CUDAShader *>(handle)->set_name(std::move(name));
                break;
            case Resource::Tag::SWAP_CHAIN:
#ifdef LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN
                reinterpret_cast<CUDASwapchain *>(handle)->set_name(std::move(name));
#endif
                break;
            case Resource::Tag::HASH_GRID:
                // todo
                break;
            default: break;
        }
    });
}

ResourceCreationInfo CUDADevice::create_hash_grid(int dim_x, int dim_y, int dim_z) noexcept {
    auto handle = hash_grid_create_device(this->handle().context(), dim_x, dim_y, dim_y);
    return {.handle = handle,
            .native_handle = nullptr};
}

void CUDADevice::destroy_hash_grid(uint64_t handle) noexcept {
    hash_grid_destroy_device(handle);
}

}// namespace luisa::compute::cuda

LUISA_EXPORT_API luisa::compute::DeviceInterface *create(luisa::compute::Context &&ctx,
                                                         const luisa::compute::DeviceConfig *config) noexcept {
    auto device_id = 0ull;
    auto binary_io = static_cast<const luisa::BinaryIO *>(nullptr);
    if (config != nullptr) {
        device_id = config->device_index;
        binary_io = config->binary_io;
        LUISA_ASSERT(!config->headless,
                     "Headless mode is not implemented yet for CUDA backend.");
    }
    return luisa::new_with_allocator<luisa::compute::cuda::CUDADevice>(
        std::move(ctx), device_id, binary_io);
}

LUISA_EXPORT_API void destroy(luisa::compute::DeviceInterface *device) noexcept {
    auto p = dynamic_cast<luisa::compute::cuda::CUDADevice *>(device);
    LUISA_ASSERT(p != nullptr, "Deleting a null CUDA device.");
    luisa::delete_with_allocator(p);
}

LUISA_EXPORT_API void backend_device_names(luisa::vector<luisa::string> &names) noexcept {
    names.clear();
    auto device_count = 0;
    luisa::compute::cuda::detail::initialize();
    LUISA_CHECK_CUDA(cuDeviceGetCount(&device_count));
    if (device_count > 0) {
        names.reserve(device_count);
        for (auto i = 0; i < device_count; i++) {
            CUdevice device{};
            LUISA_CHECK_CUDA(cuDeviceGet(&device, i));
            static thread_local char name[1024];
            LUISA_CHECK_CUDA(cuDeviceGetName(name, 1024, device));
            names.emplace_back(name);
        }
    }
}

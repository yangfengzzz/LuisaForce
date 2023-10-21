//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/magic_enum.h"
#include "runtime/command_list.h"

#include "cuda_error.h"
#include "cuda_buffer.h"
#include "cuda_stream.h"
#include "cuda_device.h"
#include "cuda_shader.h"
#include "cuda_host_buffer_pool.h"
#include "cuda_texture.h"
#include "cuda_bindless_array.h"
#include "cuda_command_encoder.h"
#include "runtime/ext/registry.h"
#include "runtime/ext/cuda/lcub/cuda_lcub_command.h"

#include "cuda_builtin_kernels/hashgrid.h"

namespace luisa::compute::cuda {

class UserCallbackContext : public CUDACallbackContext {

public:
    using CallbackContainer = CommandList::CallbackContainer;

private:
    CallbackContainer _functions;

private:
    [[nodiscard]] static auto &_object_pool() noexcept {
        static Pool<UserCallbackContext, true> pool;
        return pool;
    }

public:
    explicit UserCallbackContext(CallbackContainer &&cbs) noexcept
        : _functions{std::move(cbs)} {}

    [[nodiscard]] static auto create(CallbackContainer &&cbs) noexcept {
        return _object_pool().create(std::move(cbs));
    }

    void recycle() noexcept override {
        for (auto &&f : _functions) { f(); }
        _object_pool().destroy(this);
    }
};

void CUDACommandEncoder::commit(CommandList::CallbackContainer &&user_callbacks) noexcept {
    if (!user_callbacks.empty()) {
        _callbacks.emplace_back(
            UserCallbackContext::create(
                std::move(user_callbacks)));
    }
    if (auto callbacks = std::move(_callbacks); !callbacks.empty()) {
        _stream->callback(std::move(callbacks));
    }
}

void CUDACommandEncoder::visit(BufferUploadCommand *command) noexcept {
    auto buffer = reinterpret_cast<const CUDABuffer *>(command->handle());
    auto address = buffer->handle() + command->offset();
    auto data = command->data();
    auto size = command->size();
    with_upload_buffer(size, [&](auto upload_buffer) noexcept {
        std::memcpy(upload_buffer->address(), data, size);
        LUISA_CHECK_CUDA(cuMemcpyHtoDAsync(
            address, upload_buffer->address(),
            size, _stream->handle()));
    });
}

void CUDACommandEncoder::visit(BufferDownloadCommand *command) noexcept {
    auto buffer = reinterpret_cast<const CUDABuffer *>(command->handle());
    auto address = buffer->handle() + command->offset();
    auto data = command->data();
    auto size = command->size();
    with_download_pool_no_fallback(size, [&](auto download_buffer) noexcept {
        if (download_buffer) {
            LUISA_CHECK_CUDA(cuMemcpyDtoHAsync(
                download_buffer->address(), address,
                size, _stream->handle()));
            LUISA_CHECK_CUDA(cuMemcpyAsync(
                reinterpret_cast<CUdeviceptr>(data),
                reinterpret_cast<CUdeviceptr>(download_buffer->address()),
                size, _stream->handle()));
        } else {
            LUISA_CHECK_CUDA(cuMemcpyDtoHAsync(
                data, address, size, _stream->handle()));
        }
    });
}

void CUDACommandEncoder::visit(BufferCopyCommand *command) noexcept {
    auto src_buffer = reinterpret_cast<const CUDABuffer *>(command->src_handle())->handle() +
                      command->src_offset();
    auto dst_buffer = reinterpret_cast<const CUDABuffer *>(command->dst_handle())->handle() +
                      command->dst_offset();
    auto size = command->size();
    LUISA_CHECK_CUDA(cuMemcpyDtoDAsync(dst_buffer, src_buffer, size, _stream->handle()));
}

void CUDACommandEncoder::visit(ShaderDispatchCommand *command) noexcept {
    reinterpret_cast<CUDAShader *>(command->handle())->launch(*this, command);
}

namespace detail {

static void memcpy_buffer_to_texture(CUdeviceptr buffer, size_t buffer_offset, size_t buffer_total_size,
                                     CUarray array, PixelStorage array_storage, uint3 array_size,
                                     CUstream stream) noexcept {
    CUDA_MEMCPY3D copy{};
    auto pitch = pixel_storage_size(array_storage, make_uint3(array_size.x, 1u, 1u));
    auto height = pixel_storage_size(array_storage, make_uint3(array_size.xy(), 1u)) / pitch;
    auto full_size = pixel_storage_size(array_storage, array_size);
    LUISA_ASSERT(buffer_offset < buffer_total_size &&
                     buffer_total_size - buffer_offset >= full_size,
                 "Buffer size too small for texture copy.");
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = buffer + buffer_offset;
    copy.srcPitch = pitch;
    copy.srcHeight = height;
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = array;
    copy.WidthInBytes = pitch;
    copy.Height = height;
    copy.Depth = array_size.z;
    LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, stream));
}

}// namespace detail

void CUDACommandEncoder::visit(BufferToTextureCopyCommand *command) noexcept {
    auto mipmap_array = reinterpret_cast<CUDATexture *>(command->texture());
    auto array = mipmap_array->level(command->level());
    auto buffer = reinterpret_cast<const CUDABuffer *>(command->buffer());
    detail::memcpy_buffer_to_texture(
        buffer->handle(), command->buffer_offset(), buffer->size_bytes(),
        array, command->storage(), command->size(), _stream->handle());
}

void CUDACommandEncoder::visit(TextureUploadCommand *command) noexcept {
    auto mipmap_array = reinterpret_cast<CUDATexture *>(command->handle());
    auto array = mipmap_array->level(command->level());
    CUDA_MEMCPY3D copy{};
    auto pitch = pixel_storage_size(command->storage(), make_uint3(command->size().x, 1u, 1u));
    auto height = pixel_storage_size(command->storage(), make_uint3(command->size().xy(), 1u)) / pitch;
    auto size_bytes = pixel_storage_size(command->storage(), command->size());
    auto data = command->data();
    with_upload_buffer(size_bytes, [&](auto upload_buffer) noexcept {
        std::memcpy(upload_buffer->address(), data, size_bytes);
        copy.srcMemoryType = CU_MEMORYTYPE_HOST;
        copy.srcHost = upload_buffer->address();
        copy.srcPitch = pitch;
        copy.srcHeight = height;
        copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        copy.dstArray = array;
        copy.WidthInBytes = pitch;
        copy.Height = height;
        copy.Depth = command->size().z;
        LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
    });
}

void CUDACommandEncoder::visit(TextureDownloadCommand *command) noexcept {
    auto mipmap_array = reinterpret_cast<CUDATexture *>(command->handle());
    auto array = mipmap_array->level(command->level());
    CUDA_MEMCPY3D copy{};
    auto pitch = pixel_storage_size(command->storage(), make_uint3(command->size().x, 1u, 1u));
    auto height = pixel_storage_size(command->storage(), make_uint3(command->size().xy(), 1u)) / pitch;
    auto size_bytes = pixel_storage_size(command->storage(), command->size());
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = array;
    copy.WidthInBytes = pitch;
    copy.Height = height;
    copy.Depth = command->size().z;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstPitch = pitch;
    copy.dstHeight = height;
    with_download_pool_no_fallback(size_bytes, [&](auto download_buffer) noexcept {
        if (download_buffer) {
            copy.dstHost = download_buffer->address();
            LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
            LUISA_CHECK_CUDA(cuMemcpyAsync(
                reinterpret_cast<CUdeviceptr>(command->data()),
                reinterpret_cast<CUdeviceptr>(download_buffer->address()),
                size_bytes, _stream->handle()));
        } else {
            copy.dstHost = command->data();
            LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
        }
    });
}

void CUDACommandEncoder::visit(TextureCopyCommand *command) noexcept {
    auto src_mipmap_array = reinterpret_cast<CUDATexture *>(command->src_handle());
    auto dst_mipmap_array = reinterpret_cast<CUDATexture *>(command->dst_handle());
    auto src_array = src_mipmap_array->level(command->src_level());
    auto dst_array = dst_mipmap_array->level(command->dst_level());
    auto pitch = pixel_storage_size(command->storage(), make_uint3(command->size().x, 1u, 1u));
    auto height = pixel_storage_size(command->storage(), make_uint3(command->size().xy(), 1u)) / pitch;
    CUDA_MEMCPY3D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = src_array;
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = dst_array;
    copy.WidthInBytes = pitch;
    copy.Height = height;
    copy.Depth = command->size().z;
    LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
}

void CUDACommandEncoder::visit(TextureToBufferCopyCommand *command) noexcept {
    auto mipmap_array = reinterpret_cast<CUDATexture *>(command->texture());
    auto array = mipmap_array->level(command->level());
    CUDA_MEMCPY3D copy{};
    auto pitch = pixel_storage_size(command->storage(), make_uint3(command->size().x, 1u, 1u));
    auto height = pixel_storage_size(command->storage(), make_uint3(command->size().xy(), 1u)) / pitch;
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = array;
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = reinterpret_cast<const CUDABuffer *>(command->buffer())->handle() +
                     command->buffer_offset();
    copy.dstPitch = pitch;
    copy.dstHeight = height;
    copy.WidthInBytes = pitch;
    copy.Height = height;
    copy.Depth = command->size().z;
    LUISA_CHECK_CUDA(cuMemcpy3DAsync(&copy, _stream->handle()));
}

void CUDACommandEncoder::visit(BindlessArrayUpdateCommand *command) noexcept {
    auto bindless_array = reinterpret_cast<CUDABindlessArray *>(command->handle());
    bindless_array->update(*this, command);
}

void CUDACommandEncoder::visit(CustomCommand *command) noexcept {
    switch (command->uuid()) {
        case to_underlying(CustomCommandUUID::CUDA_LCUB_COMMAND): {
            auto lcub_command = dynamic_cast<CudaLCubCommand *>(command);
            LUISA_ASSERT(lcub_command != nullptr, "Invalid CudaLCuBCommand.");
            lcub_command->func(_stream->handle());
            break;
        }
        default:
            LUISA_ERROR_WITH_LOCATION("Custom command (UUID = 0x{:04x}) "
                                      "is not supported on CUDA.",
                                      command->uuid());
    }
}

void CUDACommandEncoder::visit(HashGridReserveCommand *command) noexcept {
    hash_grid_reserve_device(command->handle(), command->num_points(), _stream->handle());
}

void CUDACommandEncoder::visit(HashGridBuildCommand *command) noexcept {
    auto points = reinterpret_cast<const CUDABuffer *>(command->points())->handle();
    hash_grid_update_device(command->handle(), command->radius(), reinterpret_cast<wp::vec3 *>(points),
                            command->num_points(), _stream->handle());
}

}// namespace luisa::compute::cuda

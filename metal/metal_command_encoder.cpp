//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "core/pool.h"
#include "core/logging.h"
#include "runtime/rhi/pixel.h"
#include "metal_buffer.h"
#include "metal_texture.h"
#include "metal_shader.h"
#include "metal_bindless_array.h"
#include "metal_command_encoder.h"
#include "runtime/ext/registry.h"
#include "runtime/ext/metal/metal_command.h"

namespace luisa::compute::metal {

class UserCallbackContext : public MetalCallbackContext {

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

MetalCommandEncoder::MetalCommandEncoder(MetalStream *stream) noexcept
    : _stream{stream} {}

void MetalCommandEncoder::_prepare_command_buffer() noexcept {
    if (_command_buffer == nullptr) {
        auto desc = MTL::CommandBufferDescriptor::alloc()->init();
        desc->setRetainedReferences(false);
#ifndef NDEBUG
        desc->setErrorOptions(MTL::CommandBufferErrorOptionEncoderExecutionStatus);
#else
        desc->setErrorOptions(MTL::CommandBufferErrorOptionNone);
#endif
        _command_buffer = _stream->queue()->commandBuffer(desc);
        desc->release();
    }
}

MetalCommandEncoder::~MetalCommandEncoder() noexcept {
    for (auto &item : pipeline_cache_states) {
        item.second->release();
    }
    pipeline_cache_states.clear();
}

void MetalCommandEncoder::add_callback(MetalCallbackContext *cb) noexcept {
    _callbacks.emplace_back(cb);
}

MTL::CommandBuffer *MetalCommandEncoder::command_buffer() noexcept {
    _prepare_command_buffer();
    return _command_buffer;
}

MTL::CommandBuffer *MetalCommandEncoder::submit(CommandList::CallbackContainer &&user_callbacks) noexcept {
    if (!user_callbacks.empty()) {
        add_callback(UserCallbackContext::create(
            std::move(user_callbacks)));
    }
    auto callbacks = std::exchange(_callbacks, {});
    if (!callbacks.empty()) { _prepare_command_buffer(); }
    auto command_buffer = std::exchange(_command_buffer, nullptr);
    if (command_buffer != nullptr) {
        _stream->submit(command_buffer, std::move(callbacks));
    }
    return command_buffer;
}

void MetalCommandEncoder::visit(BufferUploadCommand *command) noexcept {
    _prepare_command_buffer();
    auto buffer = reinterpret_cast<const MetalBuffer *>(command->handle())->handle();
    auto offset = command->offset();
    auto size = command->size();
    auto data = command->data();
    with_upload_buffer(size, [&](MetalStageBufferPool::Allocation *upload_buffer) noexcept {
        auto p = static_cast<std::byte *>(upload_buffer->buffer()->contents()) +
                 upload_buffer->offset();
        std::memcpy(p, data, size);
        auto encoder = _command_buffer->blitCommandEncoder();
        encoder->copyFromBuffer(upload_buffer->buffer(),
                                upload_buffer->offset(),
                                buffer, offset, size);
        encoder->endEncoding();
    });
}

void MetalCommandEncoder::visit(BufferDownloadCommand *command) noexcept {
    _prepare_command_buffer();
    auto buffer = reinterpret_cast<const MetalBuffer *>(command->handle())->handle();
    auto offset = command->offset();
    auto size = command->size();
    auto data = command->data();
    with_download_buffer(size, [&](MetalStageBufferPool::Allocation *download_buffer) noexcept {
        auto encoder = _command_buffer->blitCommandEncoder();
        encoder->copyFromBuffer(buffer, offset,
                                download_buffer->buffer(),
                                download_buffer->offset(), size);
        encoder->endEncoding();
        // copy from download buffer to user buffer
        // TODO: use a better way to pass data back to CPU
        add_callback(FunctionCallbackContext::create([download_buffer, data, size] {
            std::memcpy(data, download_buffer->data(), size);
        }));
    });
}

void MetalCommandEncoder::visit(BufferCopyCommand *command) noexcept {
    _prepare_command_buffer();
    auto src_buffer = reinterpret_cast<const MetalBuffer *>(command->src_handle())->handle();
    auto dst_buffer = reinterpret_cast<const MetalBuffer *>(command->dst_handle())->handle();
    auto src_offset = command->src_offset();
    auto dst_offset = command->dst_offset();
    auto size = command->size();
    auto encoder = _command_buffer->blitCommandEncoder();
    encoder->copyFromBuffer(src_buffer, src_offset, dst_buffer, dst_offset, size);
    encoder->endEncoding();
}

void MetalCommandEncoder::visit(BufferToTextureCopyCommand *command) noexcept {
    _prepare_command_buffer();
    auto buffer = reinterpret_cast<const MetalBuffer *>(command->buffer())->handle();
    auto buffer_offset = command->buffer_offset();
    auto texture = reinterpret_cast<const MetalTexture *>(command->texture())->handle();
    auto texture_level = command->level();
    auto size = command->size();
    auto pitch_size = pixel_storage_size(command->storage(), make_uint3(size.x, 1u, 1u));
    auto image_size = pixel_storage_size(command->storage(), make_uint3(size.xy(), 1u));
    auto encoder = _command_buffer->blitCommandEncoder();
    encoder->copyFromBuffer(buffer, buffer_offset, pitch_size, image_size,
                            MTL::Size{size.x, size.y, size.z},
                            texture, 0u, texture_level,
                            MTL::Origin{0u, 0u, 0u});
    encoder->endEncoding();
}

void MetalCommandEncoder::visit(ShaderDispatchCommand *command) noexcept {
    _prepare_command_buffer();
    auto shader = reinterpret_cast<const MetalShader *>(command->handle());
    shader->launch(*this, command);
}

void MetalCommandEncoder::visit(TextureUploadCommand *command) noexcept {
    _prepare_command_buffer();
    auto texture = reinterpret_cast<const MetalTexture *>(command->handle())->handle();
    auto level = command->level();
    auto size = command->size();
    auto data = command->data();
    auto storage = command->storage();
    auto pitch_size = pixel_storage_size(command->storage(), make_uint3(size.x, 1u, 1u));
    auto image_size = pixel_storage_size(command->storage(), make_uint3(size.xy(), 1u));
    auto total_size = image_size * size.z;
    with_upload_buffer(total_size, [&](MetalStageBufferPool::Allocation *upload_buffer) noexcept {
        auto p = static_cast<std::byte *>(upload_buffer->buffer()->contents()) +
                 upload_buffer->offset();
        std::memcpy(p, data, total_size);
        auto encoder = _command_buffer->blitCommandEncoder();
        encoder->copyFromBuffer(upload_buffer->buffer(), upload_buffer->offset(),
                                pitch_size, image_size, MTL::Size{size.x, size.y, size.z},
                                texture, 0u, level, MTL::Origin{0u, 0u, 0u});
        encoder->endEncoding();
    });
}

void MetalCommandEncoder::visit(TextureDownloadCommand *command) noexcept {
    _prepare_command_buffer();
    auto texture = reinterpret_cast<const MetalTexture *>(command->handle())->handle();
    auto level = command->level();
    auto size = command->size();
    auto data = command->data();
    auto storage = command->storage();
    auto pitch_size = pixel_storage_size(command->storage(), make_uint3(size.x, 1u, 1u));
    auto image_size = pixel_storage_size(command->storage(), make_uint3(size.xy(), 1u));
    auto total_size = image_size * size.z;
    with_download_buffer(total_size, [&](MetalStageBufferPool::Allocation *download_buffer) noexcept {
        auto encoder = _command_buffer->blitCommandEncoder();
        encoder->copyFromTexture(texture, 0u, level,
                                 MTL::Origin{0u, 0u, 0u},
                                 MTL::Size{size.x, size.y, size.z},
                                 download_buffer->buffer(),
                                 download_buffer->offset(),
                                 pitch_size, image_size);
        encoder->endEncoding();
        // copy from download buffer to user buffer
        // TODO: use a better way to pass data back to CPU
        add_callback(FunctionCallbackContext::create([download_buffer, data, total_size] {
            std::memcpy(data, download_buffer->data(), total_size);
        }));
    });
}

void MetalCommandEncoder::visit(TextureCopyCommand *command) noexcept {
    _prepare_command_buffer();
    auto src_texture = reinterpret_cast<const MetalTexture *>(command->src_handle())->handle();
    auto dst_texture = reinterpret_cast<const MetalTexture *>(command->dst_handle())->handle();
    auto src_level = command->src_level();
    auto dst_level = command->dst_level();
    auto storage = command->storage();
    auto size = command->size();
    auto encoder = _command_buffer->blitCommandEncoder();
    encoder->copyFromTexture(src_texture, 0u, src_level,
                             MTL::Origin{0u, 0u, 0u},
                             MTL::Size{size.x, size.y, size.z},
                             dst_texture, 0u, dst_level,
                             MTL::Origin{0u, 0u, 0u});
    encoder->endEncoding();
}

void MetalCommandEncoder::visit(TextureToBufferCopyCommand *command) noexcept {
    _prepare_command_buffer();
    auto texture = reinterpret_cast<const MetalTexture *>(command->texture())->handle();
    auto texture_level = command->level();
    auto buffer = reinterpret_cast<const MetalBuffer *>(command->buffer())->handle();
    auto buffer_offset = command->buffer_offset();
    auto size = command->size();
    auto pitch_size = pixel_storage_size(command->storage(), make_uint3(size.x, 1u, 1u));
    auto image_size = pixel_storage_size(command->storage(), make_uint3(size.xy(), 1u));
    auto encoder = _command_buffer->blitCommandEncoder();
    encoder->copyFromTexture(texture, 0u, texture_level,
                             MTL::Origin{0u, 0u, 0u},
                             MTL::Size{size.x, size.y, size.z},
                             buffer, buffer_offset,
                             pitch_size, image_size);
    encoder->endEncoding();
}

void MetalCommandEncoder::visit(BindlessArrayUpdateCommand *command) noexcept {
    _prepare_command_buffer();
    auto bindless_array = reinterpret_cast<MetalBindlessArray *>(command->handle());
    bindless_array->update(*this, command);
}

void MetalCommandEncoder::visit(CustomCommand *command) noexcept {
    _prepare_command_buffer();

    switch (command->uuid()) {
        case to_underlying(CustomCommandUUID::CUSTOM_DISPATCH): {
            auto metal_command = dynamic_cast<MetalCommand *>(command);
            LUISA_ASSERT(metal_command != nullptr, "Invalid CudaLCuBCommand.");

            auto pso = find_pipeline_cache(metal_command->shader_source, metal_command->macros);
            auto encoder = _command_buffer->computeCommandEncoder();
            encoder->setComputePipelineState(pso);
            metal_command->func(encoder, pso->threadExecutionWidth());
            encoder->endEncoding();

            break;
        }
        default:
            LUISA_ERROR_WITH_LOCATION(
                "Custom command (uuid = 0x{:04x}) is not "
                "supported in Metal backend.",
                command->uuid());
    }
}

MTL::ComputePipelineState *MetalCommandEncoder::find_pipeline_cache(const std::string &raw_source,
                                                                    const std::unordered_map<std::string, std::string> &macros) {
    luisa::vector<NS::Object *> property_keys;
    luisa::vector<NS::Object *> property_values;

    auto hash = luisa::hash_value(raw_source);
    for (auto &item : macros) {
        property_keys.push_back(NS::String::string(item.first.c_str(), NS::UTF8StringEncoding));
        property_values.push_back(NS::String::string(item.second.c_str(), NS::UTF8StringEncoding));

        hash = luisa::hash_combine({hash, luisa::hash_value(item.first)});
        hash = luisa::hash_combine({hash, luisa::hash_value(item.second)});
    }
    auto iter = pipeline_cache_states.find(hash);
    if (iter == pipeline_cache_states.end()) {
        auto source = NS::String::string(raw_source.c_str(), NS::UTF8StringEncoding);

        NS::Error *error{nullptr};
        auto option = make_shared(MTL::CompileOptions::alloc()->init());

        NS::Dictionary *dict = NS::Dictionary::alloc()->init(property_keys.data(),
                                                             property_values.data(), macros.size())
                                   ->autorelease();
        option->setPreprocessorMacros(dict);
        auto library = make_shared(device()->newLibrary(source, option.get(), &error));
        if (error != nullptr) {
            LUISA_ERROR_WITH_LOCATION("Could not load Metal shader library: {}",
                                      error->description()->cString(NS::StringEncoding::UTF8StringEncoding));
        }

        auto functionName = NS::String::string("main", NS::UTF8StringEncoding);
        auto function = make_shared(library->newFunction(functionName));

        auto pso = device()->newComputePipelineState(function.get(), &error);
        if (error != nullptr) {
            LUISA_ERROR_WITH_LOCATION("could not create pso: {}",
                                      error->description()->cString(NS::StringEncoding::UTF8StringEncoding));
        }

        pipeline_cache_states[hash] = pso;
        return pipeline_cache_states[hash];
    } else {
        return iter->second;
    }
}

}// namespace luisa::compute::metal

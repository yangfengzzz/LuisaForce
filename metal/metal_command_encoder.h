//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "runtime/rhi/command.h"
#include "runtime/command_list.h"
#include "metal_api.h"
#include "metal_stream.h"

namespace luisa::compute::metal {

class MetalCommandEncoder final: public MutableCommandVisitor {
private:
    MetalStream *_stream;
    MTL::CommandBuffer *_command_buffer{nullptr};
    luisa::vector<MetalCallbackContext *> _callbacks;
    std::unordered_map<std::size_t, MTL::ComputePipelineState *> pipeline_cache_states{};

private:
    void _prepare_command_buffer() noexcept;

    MTL::ComputePipelineState * find_pipeline_cache(const std::string& source, const std::string& entry,
                                                   const std::unordered_map<std::string, std::string>& macros);

public:
    explicit MetalCommandEncoder(MetalStream *stream) noexcept;
    ~MetalCommandEncoder() noexcept override;
    [[nodiscard]] auto stream() const noexcept { return _stream; }
    [[nodiscard]] auto device() const noexcept { return _stream->device(); }
    [[nodiscard]] MTL::CommandBuffer *command_buffer() noexcept;
    void visit(BufferUploadCommand *command) noexcept override;
    void visit(BufferDownloadCommand *command) noexcept override;
    void visit(BufferCopyCommand *command) noexcept override;
    void visit(BufferToTextureCopyCommand *command) noexcept override;
    void visit(ShaderDispatchCommand *command) noexcept override;
    void visit(TextureUploadCommand *command) noexcept override;
    void visit(TextureDownloadCommand *command) noexcept override;
    void visit(TextureCopyCommand *command) noexcept override;
    void visit(TextureToBufferCopyCommand *command) noexcept override;
    void visit(BindlessArrayUpdateCommand *command) noexcept override;
    void visit(CustomCommand *command) noexcept override;
    void add_callback(MetalCallbackContext *cb) noexcept;
    virtual MTL::CommandBuffer *submit(CommandList::CallbackContainer &&user_callbacks) noexcept;

    template<typename F>
    void with_upload_buffer(size_t size, F &&f) noexcept {
        _prepare_command_buffer();
        auto upload_buffer = _stream->upload_pool()->allocate(size);
        f(upload_buffer);
        add_callback(upload_buffer);
    }

    template<typename F>
    void with_download_buffer(size_t size, F &&f) noexcept {
        _prepare_command_buffer();
        auto download_buffer = _stream->download_pool()->allocate(size);
        f(download_buffer);
        add_callback(download_buffer);
    }
};

}// namespace luisa::compute::metal

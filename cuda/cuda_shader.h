//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <span>
#include <memory>

#include "core/basic_types.h"
#include "core/spin_mutex.h"
#include "ast/usage.h"

namespace luisa::compute {
class ShaderDispatchCommand;
}// namespace luisa::compute

namespace luisa::compute::cuda {

class CUDACommandEncoder;

class CUDAShader {

private:
    luisa::vector<Usage> _argument_usages;
    luisa::string _name;
    mutable spin_mutex _name_mutex;

private:
    virtual void _launch(CUDACommandEncoder &encoder,
                         ShaderDispatchCommand *command) const noexcept = 0;

public:
    explicit CUDAShader(luisa::vector<Usage> arg_usages) noexcept;
    CUDAShader(CUDAShader &&) noexcept = delete;
    CUDAShader(const CUDAShader &) noexcept = delete;
    CUDAShader &operator=(CUDAShader &&) noexcept = delete;
    CUDAShader &operator=(const CUDAShader &) noexcept = delete;
    virtual ~CUDAShader() noexcept = default;
    [[nodiscard]] Usage argument_usage(size_t i) const noexcept;
    [[nodiscard]] virtual void *handle() const noexcept = 0;
    void launch(CUDACommandEncoder &encoder,
                ShaderDispatchCommand *command) const noexcept;
    void set_name(std::string &&name) noexcept;
};

}// namespace luisa::compute::cuda

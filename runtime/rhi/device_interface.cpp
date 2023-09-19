//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/rhi/device_interface.h"
#include "runtime/context.h"

namespace luisa::compute {

DeviceInterface::DeviceInterface(Context &&ctx) noexcept
    : _ctx_impl{std::move(ctx).impl()} {}

DeviceInterface::~DeviceInterface() noexcept = default;

Context DeviceInterface::context() const noexcept {
    return Context{_ctx_impl};
}

}// namespace luisa::compute

//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/ext/metal/mps_command.h"
#include "metal_buffer.h"
#include "metal_device.h"
#include "core/logging.h"

namespace luisa::compute::metal {
MPSCommand::UCommand MPSCommand::clone() {
    return luisa::make_unique<luisa::compute::metal::MPSCommand>(this);
}

MPSCommand::MPSCommand(MPSCommand *command) noexcept
    : CustomCommand{}, func{command->func}, kernel_func{command->kernel_func}, objects{command->objects} {
    for (auto &obj : objects) {
        obj->retain();
    }
}

MPSCommand::~MPSCommand() {
    for (auto &obj : objects) {
        obj->release();
    }
}
}// namespace luisa::compute::metal
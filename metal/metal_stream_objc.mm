//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <Metal/Metal.h>
#include "core/logging.h"

LUISA_EXTERN_C void luisa_compute_metal_stream_print_function_logs(id<MTLLogContainer> logs) {
    if (logs != nullptr) {
        for (id<MTLFunctionLog> log in logs) {
            LUISA_INFO("[MTLFunctionLog] {}", log.debugDescription.UTF8String);
        }
    }
}

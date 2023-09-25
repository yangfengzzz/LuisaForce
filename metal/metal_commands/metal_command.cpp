//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/ext/metal/metal_command.h"
#include "metal_buffer.h"
#include "metal_device.h"
#include "core/logging.h"

namespace luisa::compute::metal {
MTL::ComputePipelineState *MetalCommand::create_pipeline_cache(MTL::Device *device,
                                                               const std::string &raw_source, const std::string &entry,
                                                               const std::unordered_map<std::string, std::string> &macros) {
    luisa::vector<NS::Object *> property_keys;
    luisa::vector<NS::Object *> property_values;

    for (auto &item : macros) {
        property_keys.push_back(NS::String::string(item.first.c_str(), NS::UTF8StringEncoding));
        property_values.push_back(NS::String::string(item.second.c_str(), NS::UTF8StringEncoding));
    }

    auto source = NS::String::string(raw_source.c_str(), NS::UTF8StringEncoding);

    NS::Error *error{nullptr};
    auto option = make_shared(MTL::CompileOptions::alloc()->init());

    NS::Dictionary *dict = NS::Dictionary::alloc()->init(property_values.data(),
                                                         property_keys.data(), macros.size())
                               ->autorelease();
    option->setPreprocessorMacros(dict);
    auto library = make_shared(device->newLibrary(source, option.get(), &error));
    if (error != nullptr) {
        LUISA_ERROR_WITH_LOCATION("Could not load Metal shader library: {}",
                                  error->description()->cString(NS::StringEncoding::UTF8StringEncoding));
    }

    auto functionName = NS::String::string(entry.c_str(), NS::UTF8StringEncoding);
    auto function = make_shared(library->newFunction(functionName));

    auto pso = device->newComputePipelineState(function.get(), &error);
    if (error != nullptr) {
        LUISA_ERROR_WITH_LOCATION("could not create pso: {}",
                                  error->description()->cString(NS::StringEncoding::UTF8StringEncoding));
    }
    pso->retain();
    return pso;
}

MetalCommand::UCommand MetalCommand::clone() {
    return luisa::make_unique<luisa::compute::metal::MetalCommand>(this);
}

MetalCommand::MetalCommand(MetalCommand *command) noexcept
    : CustomCommand{}, func{command->func}, pso_func{command->pso_func}, pso{command->pso} {
    pso->retain();
}

MetalCommand::~MetalCommand() {
    pso->release();
}

void MetalCommand::alloc_pso(Device *device) {
    pso = pso_func(dynamic_cast<MetalDevice *>(device->impl())->handle());
}

}// namespace luisa::compute::metal
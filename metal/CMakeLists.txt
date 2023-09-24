#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

cmake_minimum_required(VERSION 3.16)

project(metal-backend LANGUAGES C CXX)

set(COMMON_FILES
        metal_bindless_array.cpp
        metal_buffer.cpp
        metal_builtin_embedded.cpp
        metal_callback_context.cpp
        metal_codegen_ast.cpp
        metal_command_encoder.cpp
        metal_compiler.cpp
        metal_debug_capture.cpp
        metal_device.cpp
        metal_event.cpp
        metal_shader.cpp
        metal_shader_metadata.cpp
        metal_stage_buffer_pool.cpp
        metal_stream_objc.mm
        metal_stream.cpp
        metal_swapchain_objc.mm
        metal_swapchain.cpp
        metal_texture.cpp
)

set(COMMANDS_FILES
        metal_commands/mad_throughput.cpp
)

source_group("common\\" FILES ${COMMON_FILES})

set(PROJECT_FILES
        ${COMMON_FILES}
        ${COMMANDS_FILES}
)

#NB: switch this to shared library and things stop working. (there is likely two copies of volk somewhere.
add_library(${PROJECT_NAME} OBJECT ${PROJECT_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../)

# Link third party libraries
target_link_libraries(${PROJECT_NAME} PUBLIC
        spdlog
        half
        magic_enum
        xxHash
        EASTL
        metal-cpp
)

if (${NEED_LINK_ATOMIC})
    target_link_libraries(${PROJECT_NAME} PUBLIC atomic)
endif ()

if (METAL_DO_CLANG_TIDY)
    set_target_properties(framework PROPERTIES CXX_CLANG_TIDY "${METAL_DO_CLANG_TIDY}")
endif ()
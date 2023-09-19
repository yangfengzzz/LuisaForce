#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

cmake_minimum_required(VERSION 3.16)

project(cuda-backend LANGUAGES C CXX)

set(COMMON_FILES
        cuda_builtin/cuda_builtin_kernels.cu
        cuda_bindless_array.cpp
        cuda_buffer.cpp
        cuda_builtin_embedded.cpp
        cuda_codegen_ast.cpp
        cuda_command_encoder.cpp
        cuda_compiler.cpp
        cuda_device.cpp
        cuda_event.cpp
        cuda_ext.cpp
        cuda_host_buffer_pool.cpp
        cuda_shader.cpp
        cuda_shader_metadata.cpp
        cuda_shader_native.cpp
        cuda_stream.cpp
        cuda_swapchain.cpp
        cuda_texture.cpp
        default_binary_io.cpp
        string_scratch.cpp
)

source_group("common\\" FILES ${COMMON_FILES})

set(PROJECT_FILES
        ${COMMON_FILES}
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
)

if (${NEED_LINK_ATOMIC})
    target_link_libraries(${PROJECT_NAME} PUBLIC atomic)
endif ()

if (METAL_DO_CLANG_TIDY)
    set_target_properties(framework PROPERTIES CXX_CLANG_TIDY "${METAL_DO_CLANG_TIDY}")
endif ()
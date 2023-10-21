#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

cmake_minimum_required(VERSION 3.16)

project(runtime LANGUAGES C CXX)

set(COMMON_FILES
        bindless_array.cpp
        buffer.cpp
        byte_buffer.cpp
        command_list.cpp
        device.cpp
        dispatch_buffer.cpp
        event.cpp
        image.cpp
        mipmap.cpp
        stream.cpp
        swapchain.cpp
        volume.cpp
        context.cpp
        hash_grid.cpp
        rhi/command_encoder.cpp
        rhi/device_interface.cpp
        rhi/pixel.cpp
        rhi/resource.cpp
)

source_group("common\\" FILES ${COMMON_FILES})

set(PROJECT_FILES
        ${COMMON_FILES}
)

#NB: switch this to shared library and things stop working. (there is likely two copies of volk somewhere.
add_library(${PROJECT_NAME} OBJECT ${PROJECT_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-deprecated)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(${PROJECT_NAME} PRIVATE /W3 /WX)
endif ()

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../)

# Link third party libraries
target_link_libraries(${PROJECT_NAME} PUBLIC
        spdlog
        half
        magic_enum
        xxHash
        EASTL
)

if (APPLE)
    target_link_libraries(${PROJECT_NAME} PUBLIC
            metal-cpp
    )
endif ()

if (${NEED_LINK_ATOMIC})
    target_link_libraries(${PROJECT_NAME} PUBLIC atomic)
endif ()

if (CMAKE_DO_CLANG_TIDY)
    set_target_properties(framework PROPERTIES CXX_CLANG_TIDY "${CMAKE_DO_CLANG_TIDY}")
endif ()
#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

if (NOT DEFINED CMAKE_C_COMPILER_LAUNCHER AND NOT DEFINED CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(CCACHE_FOUND ccache)
    find_program(SCCACHE_FOUND sccache)
    if (CCACHE_FOUND AND NOT SCCACHE_FOUND)
        message("setting CCACHE to ${CCACHE_FOUND}")
        set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_FOUND})
        set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_FOUND})
    elseif (SCCACHE_FOUND AND NOT CCACHE_FOUND)
        message("setting CCACHE to ${CCACHE_FOUND}")
        set(CMAKE_C_COMPILER_LAUNCHER ${SCCACHE_FOUND})
        set(CMAKE_CXX_COMPILER_LAUNCHER ${SCCACHE_FOUND})
    endif (CCACHE_FOUND AND NOT SCCACHE_FOUND)
endif ()

cmake_minimum_required(VERSION 3.16)

# globally add VKB_DEBUG for the debug build
add_compile_definitions($<$<CONFIG:DEBUG>:METAL_DEBUG>)

project(LuisaForce)

# Add path for local cmake scripts
list(APPEND CMAKE_MODULE_PATH
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/USDInstall)

include(global_options)
include(check_atomic)

# usd
include(pxrConfig)
include_directories(${PXR_INCLUDE_DIRS})

if (WIN32)
    add_compile_definitions(LUISA_PLATFORM_WINDOWS=1)
elseif (UNIX)
    add_compile_definitions(LUISA_PLATFORM_UNIX=1)
    if (APPLE)
        add_compile_definitions(LUISA_PLATFORM_APPLE=1)
    endif ()
endif ()

# CUDA
find_package(CUDA)
if (CUDA_FOUND)
    message("Found CUDA")
    find_package(CUDAToolkit)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
    enable_language(CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    list(APPEND CUDA_NVCC_FLAGS --std=c++20)
    add_compile_definitions(LUISA_PLATFORM_CUDA=1)

    find_package(Vulkan)
    if (VULKAN_FOUND)
        message("Found Vulkan")
        add_compile_definitions(LUISA_BACKEND_ENABLE_VULKAN_SWAPCHAIN=1)
    endif ()
endif ()

if (APPLE)
    add_definitions(-DPXR_METAL_SUPPORT_ENABLED)
endif ()

# Add third party libraries
add_subdirectory(third_party)

add_subdirectory(core)
add_subdirectory(ast)
add_subdirectory(runtime)
add_subdirectory(dsl)
add_subdirectory(gui)

if (CUDA_FOUND)
    if (VULKAN_FOUND)
        add_subdirectory(vulkan)
    endif ()
    add_subdirectory(cuda)
endif ()

if (APPLE)
    add_subdirectory(metal)
endif ()

add_subdirectory(apps)
add_subdirectory(benchmark)
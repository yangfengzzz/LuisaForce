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
        cuda_host_buffer_pool.cpp
        cuda_shader.cpp
        cuda_shader_metadata.cpp
        cuda_shader_native.cpp
        cuda_stream.cpp
        cuda_swapchain.cpp
        cuda_texture.cpp
)

set(CUB_FILES
        lcub/device_merge_sort.cpp
        lcub/device_partition.cpp
        lcub/device_radix_sort.cpp
        lcub/device_reduce.cpp
        lcub/device_run_length_encode.cpp
        lcub/device_scan.cpp
        lcub/device_select.cpp
        lcub/device_spmv.cpp
        lcub/private/dcub/device_merge_sort.cu
        lcub/private/dcub/device_partition.cu
        lcub/private/dcub/device_radix_sort.cu
        lcub/private/dcub/device_reduce.cu
        lcub/private/dcub/device_run_length_encode.cu
        lcub/private/dcub/device_scan.cu
        lcub/private/dcub/device_select.cu
        lcub/private/dcub/device_spmv.cu
)

set(CUSTOM_COMMANDS
        cuda_commands/mad_throughput.cu
        cuda_commands/matmul.cu
)

set(MATH_FILES
        math/cuda_math_utils.h
        math/initializer_array.h
        math/vec.h
        math/mat.h
        math/quat.h
        math/array.h
        math/rand.h
        math/noise.h
        math/range.h
        math/bounds.h
        math/solid_angle.h
        math/spatial.h
        # method
        math/svd.h
        math/intersect.h
        math/intersect_tri.h
        math/matnn.h
        math/fabric.h
)

set(STL_FILES
        stl/cuda_util.h
        stl/cuda_util.cpp
        stl/cuda_context.h
        stl/cuda_context.cu
        stl/cuda_context.cpp
        stl/sort.h
        stl/sort.cu
)

if (VULKAN_FOUND)
    set(VULKAN_CUDA_FILES
            cuda_semaphore.cpp
    )
endif ()

source_group("common\\" FILES ${COMMON_FILES})

set(PROJECT_FILES
        ${COMMON_FILES}
        ${VULKAN_CUDA_FILES}
        ${CUB_FILES}
        ${CUSTOM_COMMANDS}
        ${MATH_FILES}
        ${STL_FILES}
)

#NB: switch this to shared library and things stop working. (there is likely two copies of volk somewhere.
add_library(${PROJECT_NAME} OBJECT ${PROJECT_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../)

# Link third party libraries
target_link_libraries(${PROJECT_NAME} PUBLIC
        core
        CUDA::cuda_driver
        CUDA::nvrtc_static
)

if (VULKAN_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC
            vulkan-backend
    )
endif ()

if (${NEED_LINK_ATOMIC})
    target_link_libraries(${PROJECT_NAME} PUBLIC atomic)
endif ()

if (METAL_DO_CLANG_TIDY)
    set_target_properties(framework PROPERTIES CXX_CLANG_TIDY "${METAL_DO_CLANG_TIDY}")
endif ()
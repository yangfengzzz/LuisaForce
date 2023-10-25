#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

cmake_minimum_required(VERSION 3.16)

project(cuda-backend LANGUAGES C CXX)

set(COMMON_FILES
        cuda_bindless_array.cpp
        cuda_buffer.cpp
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

set(BUILTIN_FILES
        cuda_builtin/cuda_builtin.h
        cuda_builtin/cuda_warp.h
        #
        cuda_builtin/math/cuda_crt.h
        cuda_builtin/math/cuda_crt.cpp
        cuda_builtin/math/cuda_math_utils.h
        cuda_builtin/math/cuda_initializer_array.h
        cuda_builtin/math/cuda_vec.h
        cuda_builtin/math/cuda_mat.h
        cuda_builtin/math/cuda_quat.h
        cuda_builtin/math/cuda_array.h
        cuda_builtin/math/cuda_fabric.h
        cuda_builtin/math/cuda_spatial.h
        #
        cuda_builtin/math/cuda_bounds.h
        cuda_builtin/math/cuda_rand.h
        cuda_builtin/math/cuda_noise.h
        cuda_builtin/math/cuda_range.h
        cuda_builtin/math/cuda_solid_angle.h
        #
        cuda_builtin/math/cuda_intersect.h
        cuda_builtin/math/cuda_intersect_tri.h
        cuda_builtin/math/cuda_svd.h
        cuda_builtin/math/cuda_matnn.h
        #
        cuda_builtin/stl/hashgrid.h
)

set(BUILTIN_KERNELS_FILES
        cuda_builtin_kernels/cuda_builtin_kernels.h
        cuda_builtin_kernels/cuda_builtin_kernels.cu
        cuda_builtin_kernels/cuda_util.h
        cuda_builtin_kernels/cuda_util.cpp
        cuda_builtin_kernels/cuda_context.h
        cuda_builtin_kernels/cuda_context.cu
        cuda_builtin_kernels/cuda_context.cpp
        # cub
        cuda_builtin_kernels/sort.h
        cuda_builtin_kernels/sort.cu
        cuda_builtin_kernels/scan.h
        cuda_builtin_kernels/scan.cu
        cuda_builtin_kernels/hashgrid.h
        cuda_builtin_kernels/hashgrid.cu
        cuda_builtin_kernels/hashgrid.cpp
        cuda_builtin_kernels/reduce.h
        cuda_builtin_kernels/reduce.cu
        cuda_builtin_kernels/reduce.cpp
        cuda_builtin_kernels/marching.h
        cuda_builtin_kernels/marching.cu
        cuda_builtin_kernels/runlength_encode.h
        cuda_builtin_kernels/runlength_encode.cu
        # cutlass
        cuda_builtin_kernels/cutlass_gemm.h
        cuda_builtin_kernels/cutlass_gemm.cu
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
        ${BUILTIN_FILES}
        ${BUILTIN_KERNELS_FILES}
)

#NB: switch this to shared library and things stop working. (there is likely two copies of volk somewhere.
add_library(${PROJECT_NAME} OBJECT ${PROJECT_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-deprecated-declarations)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(${PROJECT_NAME} PRIVATE /W3 /WX)
endif ()

target_include_directories(${PROJECT_NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR} .
        ${CMAKE_CURRENT_SOURCE_DIR}/cuda_builtin
        ./)

# Link third party libraries
target_link_libraries(${PROJECT_NAME} PUBLIC
        core
        CUDA::cuda_driver
        CUDA::nvrtc_static
        CUTLASS
        cutlass_tools_util_includes
)

if (VULKAN_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC
            vulkan-backend
    )
endif ()

if (${NEED_LINK_ATOMIC})
    target_link_libraries(${PROJECT_NAME} PUBLIC atomic)
endif ()

if (CMAKE_DO_CLANG_TIDY)
    set_target_properties(framework PROPERTIES CXX_CLANG_TIDY "${CMAKE_DO_CLANG_TIDY}")
endif ()
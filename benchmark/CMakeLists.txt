#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

# create sample app project
project(kernel-benchmark LANGUAGES C CXX)

set(BENCH_SRC
        utils/data_type_util.cpp
        utils/status_util.cpp
        ##
        main.cpp
        mad_throughput.cpp
        matmul_tiled.cpp
        #        mmt.cpp
        #        atomic_reduce.cpp
        #        one_workgroup_reduce.cpp
        #        tree_reduce.cpp
        #        simd_group_arithmetic.cpp
)

set(CUDA_SRC
        matmul_tiled.cu)

set(SRC
        ${BENCH_SRC}
)

if (CUDA_FOUND)
    list(APPEND SRC ${CUDA_SRC})
endif ()

add_executable(${PROJECT_NAME} ${SRC})

target_link_libraries(${PROJECT_NAME}
        core
        ast
        dsl
        gui
        runtime
        benchmark
        gtest
)

if (APPLE)
    target_link_libraries(${PROJECT_NAME}
            metal-backend
    )
endif ()

if (CUDA_FOUND)
    if (VULKAN_FOUND)
        target_link_libraries(${PROJECT_NAME}
                vulkan-backend
        )
    endif ()

    target_link_libraries(${PROJECT_NAME}
            cuda-backend
    )
endif ()

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../)

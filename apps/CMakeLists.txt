#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

cmake_minimum_required(VERSION 3.10)
project(apps)

add_executable(${PROJECT_NAME}
        mpm88_app.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../)

set(PXR_LIBRARY_NAMES)
foreach (_item ${PXR_LIBRARIES}) # Iterate over list of absolute paths
    get_filename_component(_t ${_item} NAME)
    list(APPEND PXR_LIBRARY_NAMES ${_t})
endforeach ()

target_link_libraries(${PROJECT_NAME}
        core
        ast
        dsl
        gui
        runtime
)

if (APPLE)
    target_link_libraries(${PROJECT_NAME}
            metal-backend
    )
endif ()

if (VULKAN_FOUND)
    target_link_libraries(${PROJECT_NAME}
            vulkan-backend
    )
endif ()

if (CUDA_FOUND)
    target_link_libraries(${PROJECT_NAME}
            cuda-backend
    )
endif ()
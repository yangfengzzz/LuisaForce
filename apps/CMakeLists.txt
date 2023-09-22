#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

# Get all project dir
FILE(GLOB sample_projects ${CMAKE_CURRENT_SOURCE_DIR}/*)

# For each project dir, build a target
FOREACH (project ${sample_projects})
    IF (IS_DIRECTORY ${project})
        # Get project name and all sources
        get_filename_component(project-name ${project} NAME)
        FILE(GLOB ${project}-src ${project}/*.cpp)

        # Create executable and link target
        add_executable(${project-name} ${${project}-src})

        target_link_libraries(${project-name}
                core
                ast
                dsl
                gui
                runtime
        )

        if (APPLE)
            target_link_libraries(${project-name}
                    metal-backend
            )
        endif ()

        if (CUDA_FOUND)
            if (VULKAN_FOUND)
                target_link_libraries(${project-name}
                        vulkan-backend
                )
            endif ()

            target_link_libraries(${project-name}
                    cuda-backend
            )
        endif ()

        message(STATUS "Adding ${project-name}")
    ENDIF ()
ENDFOREACH ()
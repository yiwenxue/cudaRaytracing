set (INCLUDE
    # common utility 
    common/CudaArray.h
    common/CudaMath.h
    common/CudaUtils.h

    # 
    interop.h

    # implementations
    camera.h
    renderer.h
)

set (SOURCE
    #
    interop.cpp

    # implementations
    renderer.cpp

    # entry
    main.cpp
)

add_executable(main ${SOURCE} ${INCLUDE})

target_include_directories(main
PUBLIC
    ./
    ${OPENGL_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/external
)

target_link_libraries(main 
PUBLIC
    imgui
    stb
    glfw
    glad
    cuda
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
)

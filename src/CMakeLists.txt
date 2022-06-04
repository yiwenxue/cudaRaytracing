set (INCLUDE
    # external 
    stb_image.h
    stb_image_write.h

    # common utility 
    common/CudaArray.h
    common/CudaMath.h
    common/CudaUtils.h

    # implementations
    camera.h
    ray.h
    sphere.h
    hittable.h
    material.h
    hittable_list.h
)

set (SOURCE
    # externals 
    stb.cpp

    # implementations

    # entry
    main.cpp
    main.cu
)

add_executable(main ${SOURCE} ${INCLUDE})

target_include_directories(main
PUBLIC
    ./
    ${CUDA_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(main 
PUBLIC
    ${CUDA_LIBRARIES}
)

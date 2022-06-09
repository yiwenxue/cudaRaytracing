set(IMGUI_HEADERS
    imgui/imgui.h
)

set(IMGUI_SOURCES
    imgui/imgui.cpp
    imgui/imgui_draw.cpp 
    imgui/imgui_demo.cpp
    imgui/imgui_tables.cpp 
    imgui/imgui_widgets.cpp
    imgui/backends/imgui_impl_glfw.cpp 
    imgui/backends/imgui_impl_opengl3.cpp
)

add_library(imgui SHARED ${IMGUI_HEADERS} ${IMGUI_SOURCES})

target_include_directories(imgui 
PUBLIC
    ./imgui
    ${GLFW_INCLUDE_DIR}
)

target_link_libraries(imgui
PUBLIC
    ${OpenGL_LIBRARIES}
    ${GLFW_LIBRARY}
)

set(STB_HEADERS
    stb/stb_image.h
    stb/stb_image_write.h
)

set(STB_SOURCES
    stb/stb.cpp
)

add_library(stb SHARED ${STB_HEADERS} ${STB_SOURCES})

target_include_directories(stb 
PUBLIC
    ./stb
)

target_link_libraries(stb
PUBLIC
    ${OpenGL_LIBRARIES}
)

add_library(glad "glad/glad.c")
target_include_directories(glad PUBLIC ./)

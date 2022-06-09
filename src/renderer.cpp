#include "renderer.h"
#include "imgui/imgui.h"

#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"

#include <string>

void glfwWindowResizeCallBack(GLFWwindow *window, int width, int height)
{
    glViewport(0, 0, width, height);
};

void glfwKeyCalback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, true);
    }
};

Renderer::Renderer() : Renderer(std::string("main"), 100, 100)
{}

Renderer::Renderer(std::string name, uint32_t width, uint32_t height) :
    _name(name), _width(width), _height(height), _size(height * width)
{
    initWindow();
    initGui();

    initRenderbuffer();
}

Renderer::~Renderer()
{
    destroyRenderbuffer();
    destroyWindow();
}

int Renderer::run(int argc, char **argv)
{
    int width, height;
    while (!glfwWindowShouldClose(_window))
    {
        // clear frame
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glfwGetWindowSize(_window, &width, &height);
        _width  = width;
        _height = height;

        // draw();

        drawGui();

        glfwSwapBuffers(_window);
        glfwPollEvents();
    }
    return 0;
}

void Renderer::initWindow()
{
    // set error callback

    if (!glfwInit())
    {
        assert("glfw init failed.");
    }

    glfwWindowHint(GLFW_DEPTH_BITS, 0);
    glfwWindowHint(GLFW_STENCIL_BITS, 0);

    glfwWindowHint(GLFW_RED_BITS, 32);
    glfwWindowHint(GLFW_GREEN_BITS, 32);
    glfwWindowHint(GLFW_BLUE_BITS, 32);
    glfwWindowHint(GLFW_ALPHA_BITS, 32);
    glfwWindowHint(GLFW_ALPHA_BITS, 32);
    glfwWindowHint(GLFW_ALPHA_BITS, 0);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

    _window = glfwCreateWindow(static_cast<int>(_width), static_cast<int>(_height), _name.c_str(),
                               nullptr, nullptr);
    assert(_window != nullptr && "GlfwWindow count be created properly\n");
    glfwShowWindow(_window);

    glfwSetWindowSizeCallback(_window, glfwWindowResizeCallBack);
    glfwSetKeyCallback(_window, glfwKeyCalback);

    glfwMakeContextCurrent(_window);
}

void Renderer::destroyWindow()
{
    assert(_window != nullptr && "GlfwWindow is destroied before desctruction\n");
    glfwDestroyWindow(_window);
    glfwTerminate();
}

void Renderer::initRenderbuffer()
{
    // init gl
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress))
    {
        assert("Failed to initialize GLAD");
    }

    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_FALSE);

    glCreateRenderbuffers(1, &_bufferObj);
    glCreateFramebuffers(1, &_fbo);
    glNamedFramebufferRenderbuffer(_fbo, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, _bufferObj);
}

void Renderer::destroyRenderbuffer()
{
    glDeleteFramebuffers(1, &_fbo);
    glDeleteRenderbuffers(1, &_bufferObj);
}

void Renderer::initGui()
{
    // init imgui
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void) io;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(_window, true);
    ImGui_ImplOpenGL3_Init();
}

void Renderer::draw()
{
    glBlitNamedFramebuffer(_fbo, 0, 0, 0, _width, _height, 0, 0, _width, _height,
                           GL_COLOR_BUFFER_BIT, GL_NEAREST);
}

void Renderer::drawGui()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    {
        ImGui::ShowDemoWindow();
    }
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

#include "renderer.h"
#include "imgui/imgui.h"

#include "imgui/backends/imgui_impl_glfw.h"
#include "imgui/backends/imgui_impl_opengl3.h"

#include "stb_image_write.h"

#include "cudaRenderer.h"

#include <string>

void glfwWindowResizeCallBack(GLFWwindow *window, int width, int height)
{
    glViewport(0, 0, width, height);
};

static void glfwErrorCallback(int error, const char *description)
{
    fputs(description, stderr);
}

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

    initRenderbuffer();

    initGui();

    cudaRenderInit();
}

Renderer::~Renderer()
{
    cudaRenderClean();
    destroyRenderbuffer();
    destroyWindow();
}

int Renderer::run(int argc, char **argv)
{
    int width, height;
    while (!glfwWindowShouldClose(_window))
    {
        framecount++;
        // clear frame
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glfwGetWindowSize(_window, &width, &height);
        _width  = width;
        _height = height;

        draw();

        drawGui();

        glfwSwapBuffers(_window);
        glfwPollEvents();
    }
    return 0;
}

void Renderer::initWindow()
{
    if (!glfwInit())
    {
        assert("glfw init failed.");
    }

    glfwWindowHint(GLFW_DEPTH_BITS, 0);
    glfwWindowHint(GLFW_STENCIL_BITS, 0);

    glfwWindowHint(GLFW_RED_BITS, 8);
    glfwWindowHint(GLFW_GREEN_BITS, 8);
    glfwWindowHint(GLFW_BLUE_BITS, 8);
    glfwWindowHint(GLFW_ALPHA_BITS, 0);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

    _window = glfwCreateWindow(static_cast<int>(_width), static_cast<int>(_height), _name.c_str(),
                               nullptr, nullptr);
    assert(_window != nullptr && "GlfwWindow count be created properly\n");
    glfwShowWindow(_window);
    glfwMakeContextCurrent(_window);
    // glfwSwapInterval(0);

    glfwSetErrorCallback(glfwErrorCallback);
    glfwSetFramebufferSizeCallback(_window, glfwWindowResizeCallBack);
    glfwSetKeyCallback(_window, glfwKeyCalback);
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

    _bufferWidth  = _width;
    _bufferHeight = _height;

    glCreateRenderbuffers(1, &_bufferObj);
    glCreateFramebuffers(1, &_fbo);
    glNamedFramebufferRenderbuffer(_fbo, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, _bufferObj);
    glNamedRenderbufferStorage(_bufferObj, GL_RGBA8, _bufferWidth, _bufferHeight);

    cudaGraphicsGLRegisterImage(&_buffer, _bufferObj, GL_RENDERBUFFER,
                                cudaGraphicsRegisterFlagsSurfaceLoadStore |
                                    cudaGraphicsRegisterFlagsWriteDiscard);

    cudaGraphicsMapResources(1, &_buffer, 0);

    cudaGraphicsSubResourceGetMappedArray(&_array, _buffer, 0, 0);

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = _array;
    cudaCreateSurfaceObject(&cudaSurf, &resDesc);

    cudaGraphicsUnmapResources(1, &_buffer, 0);

    cudaMalloc((void **) &_cuda_res, sizeof(float4) * _bufferWidth * _bufferHeight);
    cudaMemset(_cuda_res, 0, sizeof(float4) * _bufferWidth * _bufferHeight);
}

void Renderer::destroyRenderbuffer()
{
    if (_cuda_res != nullptr)
    {
        cudaFree(_cuda_res);
    }
    if (_buffer != nullptr)
    {
        cudaGraphicsUnregisterResource(_buffer);
    }

    cudaDestroySurfaceObject(cudaSurf);

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
    cudaGraphicsMapResources(1, &_buffer, 0);

    // cudaGraphicsSubResourceGetMappedArray(&_array, _buffer, 0, 0);

    cudaRenderDraw();

    cudaGraphicsUnmapResources(1, &_buffer, 0);

    glBlitNamedFramebuffer(_fbo, 0, 0, 0, _bufferWidth, _bufferHeight, 0, _height, _width, 0,
                           GL_COLOR_BUFFER_BIT, GL_LINEAR);
}

float getter(void *data, int idx)
{
    return ((float *) data)[idx % 60];
}

void Renderer::drawGui()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    {
        ImGui::Begin("cuda render");

        float max_scale = 0;
        for (int i = 0; i < 60; i++)
        {
            if (frame_time[i] > max_scale)
                max_scale = frame_time[i];
        }

        ImGui::Text("average frame time: %.2f (ms)", total_time / 60.0);
        ImGui::Text("average frame rate: %.2f (fps)", 1000 / total_time * 60.0);

        ImGui::PlotLines("statics", getter, frame_time, 60, framecount, "frame time (ms)", 0,
                         max_scale, {200, 60});

        ImGui::Text("frame count: #%d", framecount);

        if (ImGui::Button("Clear"))
        {
            _needClear = 1;
            cudaRenderClear();
            framecount = 0;
        }

        if (ImGui::Button("Save"))
        {
            saveFig();
        }

        ImGui::End();
    }
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Renderer::cudaRenderClear()
{
    cudaMemset(_cuda_res, 0, sizeof(float4) * _bufferWidth * _bufferHeight);
}

void Renderer::saveFig()
{
    int            size   = _bufferWidth * _bufferHeight;
    unsigned char *pixels = new unsigned char[size * 4];

    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, _bufferWidth, _bufferHeight, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    std::string filename = "./output_" + std::to_string(framecount) + ".png";

    stbi_write_png(filename.c_str(), _bufferWidth, _bufferHeight, 4, pixels, _bufferWidth * 4);
    delete pixels;
}

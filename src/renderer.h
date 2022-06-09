#pragma once

#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>

#include <string>

#define GET_PROC_ADDRESS(str) glXGetProcAddress((const GLubyte *) str)

class Renderer
{
public:
    Renderer();

    Renderer(std::string name, uint32_t width, uint32_t height);

    virtual ~Renderer();

    inline uint32_t getSize() const
    {
        return _size;
    }
    inline uint32_t getWidth() const
    {
        return _width;
    }
    inline uint32_t getHeight() const
    {
        return _height;
    }

    virtual int run(int argc, char **argv);

    virtual void draw();
    virtual void drawGui();

protected:
    void initWindow();
    void initRenderbuffer();

    void destroyWindow();
    void destroyRenderbuffer();

    void initGui();

    GLuint                _bufferObj{0};
    cudaGraphicsResource *_buffer{nullptr};
    GLuint                _fbo{0};

    uint32_t _width{100};
    uint32_t _height{100};

    uint32_t _size{10000};

    std::string _name{"main window"};

private:
    GLFWwindow *_window;
};

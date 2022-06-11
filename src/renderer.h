#pragma once

#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

#include <string>

#define GET_PROC_ADDRESS(str) glXGetProcAddress((const GLubyte *) str)

struct HitRecord;
struct HitTable;
struct Camera;

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

    inline uint32_t getBufferSize() const
    {
        return _bufferWidth * _bufferHeight;
    }
    inline uint32_t getBufferWidth() const
    {
        return _bufferWidth;
    }
    inline uint32_t getBufferHeight() const
    {
        return _bufferHeight;
    }

    virtual int run(int argc, char **argv);

    virtual void draw();
    virtual void drawGui();

    virtual void cudaRenderInit();
    virtual void cudaRenderDraw();
    virtual void cudaRenderClear();
    virtual void cudaRenderClean();

protected:
    void initWindow();
    void initRenderbuffer();

    void destroyWindow();
    void destroyRenderbuffer();

    void initGui();

    void changeSize();

    void saveFig();

    GLuint _bufferObj{0};
    GLuint _fbo{0};

    cudaGraphicsResource *_buffer{nullptr};
    cudaArray            *_array{nullptr};
    float4               *_cuda_res{nullptr};

    GLuint glProgram;

    uint32_t _width{100};
    uint32_t _height{100};

    uint32_t _bufferWidth;
    uint32_t _bufferHeight;

    uint32_t _size{10000};

    std::string _name{"main window"};

    bool _needClear{true};

    void               *d_rand_state;
    void               *d_rand_state2;
    HitTable          **d_list;
    HitTable          **d_world;
    Camera            **d_camera;
    cudaSurfaceObject_t cudaSurf = 0;

    float frame_time[60] = {};
    float total_time{0};

    uint framecount = 0;

private:
    GLFWwindow *_window;
};

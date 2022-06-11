#include <iostream>

#include "cudaRenderer.h"
#include "renderer.h"

#define width 1920
#define height 1080

int main(int argc, char **argv)
{
    // ray_tracing();
    std::string name("hello world");
    Renderer    render(name, width, height);

    render.run(argc, argv);

    std::cout << "Exit" << std::endl;

    return 0;
}

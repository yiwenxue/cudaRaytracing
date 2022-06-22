#include <iostream>

#include "cudaRenderer.h"
#include "renderer.h"

#include "primitive.h"
#include "scene.h"

#define width 1920
#define height 1080

void ray_tracing_demo(int argc, char **argv)
{
    std::string name("hello world");
    Renderer    renderer(name, width, height);
    renderer.run(argc, argv);
    std::cout << "Exit" << std::endl;
}

int main(int argc, char **argv)
{
    // ray tracing demo
    // ray_tracing_demo(argc, argv);

    // spatial acceleration demo
    build_basic_scene();

    char buffer[2048];

    auto *scene = Scene::getInstance();

    auto shapes = scene->getShapes();

    scene->buildPrimitive();

    scene->buildAggregate();

    // for (auto *shape : shapes)
    // {
    //     shape->toString(buffer);
    //     printf("%s\n", buffer);
    // }

    delete scene;
    return 0;
}

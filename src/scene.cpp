#include "scene.h"

Scene *Scene::instance = nullptr;

void Scene::addShape(Shape *shape)
{
    shapes.push_back(shape);
}

void Scene::addLight()
{}

void Scene::addCamera()
{}

void Scene::buildAggregate()
{
    if (bvh_aggregate)
    {
        bvh_aggregate.reset();
    }
    bvh_aggregate = std::make_unique<BVHAggregate>(primitives, 5);
}

void Scene::buildPrimitive()
{
    for (auto *shape : shapes)
    {
        primitives.push_back(new SimplePrimitive(shape, nullptr));
    }
}

Sphere *random_Sphare()
{
    Vec3f position = Vec3f(rnd(100.), rnd(100.), rnd(100.));
    float radius   = rnd(10.);
    return new Sphere(position, radius);
}

void build_basic_scene()
{
    Scene *scene = Scene::getInstance();

    for (int i = 0; i < 100; i++)
    {
        scene->addShape(random_Sphare());
    }
}

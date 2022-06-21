#pragma once

// #include "material.h"
#include "aggregates.h"
#include "primitive.h"
#include "shape.h"

#include <memory>
#include <vector>

class Scene
{
    static Scene *instance;

public:
    static Scene *getInstance()
    {
        if (instance == nullptr)
        {
            instance = new Scene();
        }
        return instance;
    }

    ~Scene()
    {
        for (auto *shape : shapes)
        {
            delete shape;
        }
        for (auto *primitive : primitives)
        {
            delete primitive;
        }
    }

    virtual void addShape(Shape *shape);
    virtual void addLight();
    virtual void addCamera();
    virtual void buildAggregate();
    virtual void buildPrimitive();

    inline std::vector<Shape *> getShapes()
    {
        return shapes;
    }

protected:
    Scene()
    {}

protected:
    std::vector<Primitive *> primitives{};
    std::vector<Shape *>     shapes{};
    // std::vector<Material>  materials{};

    std::unique_ptr<BVHAggregate> bvh_aggregate;
};

void build_basic_scene();

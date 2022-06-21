#pragma once
#include "common/CudaMath.h"
#include "shape.h"

struct Shape;
struct Material;

class Primitive
{
public:
    CPU_GPU Primitive()
    {}
    CPU_GPU virtual ~Primitive()
    {}
    CPU_GPU virtual bool intersect(const Ray &ray, float tMax, Intersection &res) const = 0;
    CPU_GPU virtual bool intersectP(const Ray &ray, float tMax = FLT_MAX) const         = 0;

    CPU_GPU virtual AABB getBounds() const = 0;

protected:
};

class SimplePrimitive : public Primitive
{
public:
    CPU_GPU SimplePrimitive(Shape *shape, Material *material) : shape(shape), material(material)
    {}

    CPU_GPU virtual ~SimplePrimitive(){};

    CPU_GPU virtual bool intersect(const Ray &ray, float tMax, Intersection &res) const override;
    CPU_GPU virtual bool intersectP(const Ray &ray, float tMax = FLT_MAX) const override;
    CPU_GPU virtual AABB getBounds() const override;

private:
    Shape    *shape;
    Material *material;
};

// class GeometricPrimitive : public Primitive
// {
// public:
//     CPU_GPU GeometricPrimitive(Shape *shape, Material *material)
//     {
//         this->shape    = shape;
//         this->material = material;
//     }

//     CPU_GPU virtual ~GeometricPrimitive(){};

//     virtual bool intersect(const Ray &ray, float tMax, Intersection &res) const override;
//     virtual bool intersectP(const Ray &ray, float tMax = FLT_MAX) const override;

// protected:
//     Shape    *shape;
//     Material *material;
// };

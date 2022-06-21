#include "primitive.h"

CPU_GPU bool SimplePrimitive::intersect(const Ray &ray, float tMax, Intersection &res) const
{
    return shape->intersect(ray, tMax, res);
}

CPU_GPU bool SimplePrimitive::intersectP(const Ray &ray, float tMax) const
{
    return shape->intersectP(ray, tMax);
}

CPU_GPU AABB SimplePrimitive::getBounds() const
{
    return shape->getBounds();
}

#include "shape.h"
#include <cstdio>

CPU_GPU bool AABB::intersectP(const Ray &ray, float tMax) const
{
    float tMin  = 0.0f;
    float tMax_ = tMax;
    for (int i = 0; i < 3; i++)
    {
        float invD = ray.invDir[i];
        float t0   = (pMin[i] - ray.origin[i]) * invD;
        float t1   = (pMax[i] - ray.origin[i]) * invD;
        if (invD < 0.0f)
        {
            std::swap(t0, t1);
        }
        tMin  = t0 > tMin ? t0 : tMin;
        tMax_ = t1 < tMax_ ? t1 : tMax_;
        if (tMin > tMax_)
        {
            return false;
        }
    }
    return true;
}

// use std::optional to make the implementation of intersectP amd intersect simpler

CPU_GPU bool IntersectTriangle(const Ray &ray, float tMax, Vec3f p0, Vec3f p1, Vec3f p2,
                               Intersection &res)
{
    // TODO (yiwenxue) : implement this function
    return false;
}

CPU_GPU bool IntersectTriangleP(const Ray &ray, float tMax, Vec3f p0, Vec3f p1, Vec3f p2)
{
    // TODO (yiwenxue) : implement this function
    return false;
}

// bool Triangle::intersect(const Ray &ray, float tMax, Intersection &res) const
// {
//     return IntersectTriangle(ray, tMax, v[0], v[1], v[2], res);
// }

// bool Triangle::intersectP(const Ray &ray, float tMax) const
// {
//     return IntersectTriangleP(ray, tMax, v[0], v[1], v[2]);
// }

bool IntersectSphere(const Ray &ray, float tMax, Vec3f center, float radius, Intersection &res)
{
    Vec3f oc     = ray.origin - center;
    float a      = ray.direct.lengthsq();
    float half_b = dot(oc, ray.direct);
    float c      = oc.lengthsq() - radius * radius;

    float discriminant = half_b * half_b - a * c;
    if (discriminant < 0)
        return false;
    float sqrtd = sqrtf(discriminant);
    float root  = (-half_b - sqrtd) / a;
    if (root < 0 || tMax < root)
    {
        root = (-half_b + sqrtd) / a;
        if (root < 0 || tMax < root)
            return false;
    }

    res.point  = ray.at(root);
    res.normal = Vec3f::normal(res.point - center);
    res.t      = root;

    return true;
}

bool IntersectSphereP(const Ray &ray, float tMax, Vec3f center, float radius)
{
    // TODO (yiwenxue) : implement this function
    return false;
}

bool Sphere::intersect(const Ray &ray, float tMax, Intersection &res) const
{
    return IntersectSphere(ray, tMax, center, radius, res);
}

bool Sphere::intersectP(const Ray &ray, float tMax) const
{
    return IntersectSphereP(ray, tMax, center, radius);
}

CPU_ONLY void Sphere::toString(char *buffer) const
{
    sprintf(buffer, "Sphere(%f, %f, %f, %f)", center.x, center.y, center.z, radius);
}

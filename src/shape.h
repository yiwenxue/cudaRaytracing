#pragma once

#include "common.h"

#include "ray.h"

#include "common/CudaMath.h"

#include <cassert>

struct Bounds
{
    CPU_GPU virtual inline float getVolume() const           = 0;
    CPU_GPU virtual inline Vec3f getCenter() const           = 0;
    CPU_GPU virtual inline Vec3f getCorner(int corner) const = 0;
};

struct Intersection
{
    Vec3f point  = {0};
    Vec3f normal = {0};
    float t      = 0;
};

struct AABB : public Bounds
{
    Vec3f aabb[2] = {0};

    CPU_GPU inline float getVolume() const
    {
        Vec3f d = aabb[1] - aabb[0];
        return d.x * d.y * d.z;
    }

    CPU_GPU inline Vec3f getCenter() const
    {
        return (aabb[0] + aabb[1]) * 0.5f;
    }

    CPU_GPU inline Vec3f getCorner(int corner) const
    {
        assert(corner >= 0 && corner < 8 && "Invalid corner index");
        return Vec3f((corner & 1) ? aabb[1].x : aabb[0].x, (corner & 2) ? aabb[1].y : aabb[0].y,
                     (corner & 4) ? aabb[1].z : aabb[0].z);
    }

    CPU_GPU inline Vec3f diagonal() const
    {
        return aabb[1] - aabb[0];
    }

    CPU_GPU inline uint maxDimension() const
    {
        Vec3f d = diagonal();
        if (d.x > d.y && d.x > d.z)
            return 0;
        else if (d.y > d.z)
            return 1;
        else
            return 2;
    }
};

CPU_GPU inline AABB Union(const AABB &a, const Vec3f &b)
{
    AABB c;
    c.aabb[0] =
        Vec3f(std::min(a.aabb[0].x, b.x), std::min(a.aabb[0].y, b.y), std::min(a.aabb[0].z, b.z));
    c.aabb[1] =
        Vec3f(std::max(a.aabb[1].x, b.x), std::max(a.aabb[1].y, b.y), std::max(a.aabb[1].z, b.z));
    return c;
}

CPU_GPU inline AABB Union(const AABB &a, const AABB &b)
{
    AABB c;
    c.aabb[0] = Vec3f(std::min(a.aabb[0].x, b.aabb[0].x), std::min(a.aabb[0].y, b.aabb[0].y),
                      std::min(a.aabb[0].z, b.aabb[0].z));
    c.aabb[1] = Vec3f(std::max(a.aabb[1].x, b.aabb[1].x), std::max(a.aabb[1].y, b.aabb[1].y),
                      std::max(a.aabb[1].z, b.aabb[1].z));
    return c;
}

struct Shape
{
    AABB bound;

    CPU_GPU virtual bool intersect(const Ray &ray, float tMax, Intersection &res) const = 0;

    CPU_GPU virtual bool intersectP(const Ray &ray, float tMax = FLT_MAX) const = 0;

    CPU_ONLY virtual void toString(char *buffer) const = 0;

    CPU_GPU AABB getBounds() const
    {
        return bound;
    }
};

// struct Triangle : public Shape
// {
//     Vec3f v[3] = {0};

//     CPU_GPU bool intersect(const Ray &ray, float tMax, Intersection &res) const override;
//     CPU_GPU bool intersectP(const Ray &ray, float tMax = FLT_MAX) const;
// };

struct Sphere : public Shape
{
    Vec3f center{0};
    float radius{0};

    CPU_GPU Sphere(Vec3f c, float r) : center(c), radius(r)
    {
        bound.aabb[0] = c - Vec3f(r);
        bound.aabb[1] = c + Vec3f(r);
    };

    CPU_GPU bool intersect(const Ray &ray, float tMax, Intersection &res) const override;
    CPU_GPU bool intersectP(const Ray &ray, float tMax = FLT_MAX) const;

    CPU_ONLY void toString(char *buffer) const override;
};

// struct Cube : public Shape
// {};

// struct Mesh : public Shape
// {
//     int     numTriangles = 0;
//     float3 *vertices;

//     CPU_GPU bool intersect(const Ray &ray, float tMax, Intersection &res) const override;
//     CPU_GPU bool intersectP(const Ray &ray, float tMax = FLT_MAX) const;
// };

// struct TriangleMesh : public Mesh
// {};

// struct QuadMesh : public Mesh
// {};

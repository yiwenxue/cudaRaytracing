#pragma once

#include "common.h"
#include "common/CudaMath.h"
#include "ray.h"

class Material;

struct HitRecord
{
    Vec3f     p;
    Vec3f     normal;
    Material *mat_ptr;
    double    t;
    bool      front_face;

    CPU_GPU void set_face_normal(const Ray &r, const Vec3f &outward_normal)
    {
        front_face = dot(r.direct, outward_normal) < 0;
        normal     = front_face ? outward_normal : outward_normal * -1.f;
    }
};

struct HitTable
{
    CPU_GPU virtual bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const = 0;
};

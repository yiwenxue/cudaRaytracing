#pragma once

#include <cuda_runtime.h>

#include "common/CudaMath.h"

#include "hittable.h"

#include "ray.h"

struct Sphere : public HitTable
{
    float     radius;
    Vec3f     center;
    Material *mat_ptr;

    __device__ Sphere(){};

    __device__ Sphere(Vec3f c, float r, Material *m) : center(c), radius(r), mat_ptr(m){};

    __device__ bool hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const override
    {
        Vec3f oc     = r.origin - center;
        float a      = r.direct.lengthsq();
        float half_b = dot(oc, r.direct);
        float c      = oc.lengthsq() - radius * radius;

        float discriminant = half_b * half_b - a * c;
        if (discriminant < 0)
            return false;
        float sqrtd = sqrtf(discriminant);
        float root  = (-half_b - sqrtd) / a;
        if (root < t_min || t_max < root)
        {
            root = (-half_b + sqrtd) / a;
            if (root < t_min || t_max < root)
                return false;
        }

        rec.t                = root;
        rec.p                = r.at(rec.t);
        Vec3f outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat_ptr = mat_ptr;

        return true;
    }
};

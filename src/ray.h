#pragma once

#include <cuda_runtime.h>

#include "common/CudaMath.h"

#include "common.h"

struct Ray
{
    Vec3f origin;
    Vec3f direct;
    Vec3f invDir;
    float tm;

    CPU_GPU Ray(){};

    CPU_GPU Ray(Vec3f ori, Vec3f dir, float tm) : origin(ori), direct(dir), tm(tm)
    {
        invDir = Vec3f(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
    };
    CPU_GPU Ray(Vec3f ori, Vec3f dir) : Ray(ori, dir, 0){};

    CPU_GPU Vec3f at(float t) const
    {
        return origin + direct * t;
    }
};

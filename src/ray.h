#pragma once

#include <cuda_runtime.h>

#include "common/CudaMath.h"

struct Ray
{
    Vec3f origin;
    Vec3f direct;
    float tm;

    __device__ Ray(){};

    __device__ Ray(Vec3f ori, Vec3f dir, float tm) : origin(ori), direct(dir), tm(tm){};
    __device__ Ray(Vec3f ori, Vec3f dir) : Ray(ori, dir, 0){};

    __device__ Vec3f at(float t) const
    {
        return origin + direct * t;
    }
};

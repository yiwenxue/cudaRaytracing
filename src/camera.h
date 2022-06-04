#pragma once
#include "ray.h"

#include "common/CudaArray.h"
#include "common/CudaMath.h"

#include <curand_kernel.h>

__device__ Vec3f random_in_unit_disk(curandState *local_rand_state)
{
    Vec3f p;
    do
    {
        p.x = curand_uniform(local_rand_state) * 2.0 - 1.0;
        p.y = curand_uniform(local_rand_state) * 2.0 - 1.0;
    } while (p.x * p.x + p.y * p.y >= 1.0f);
    return p;
}

class Camera
{
public:
    __device__ Camera(Vec3f lookfrom, Vec3f lookat, Vec3f vup, float vfov, float aspect,
                      float aperture, float focus_dist)
    {
        lens_radius       = aperture / 2.0f;
        float theta       = vfov * ((float) M_PI) / 180.0f;
        float half_height = tan(theta / 2.0f);
        float half_width  = aspect * half_height;
        origin            = lookfrom;
        w                 = Vec3f::normal(lookfrom - lookat);
        u                 = Vec3f::normal(Vec3f::cross(vup, w));
        v                 = cross(w, u);
        lower_left_corner =
            origin - u * half_width * focus_dist - v * half_height * focus_dist - w * focus_dist;
        horizontal = u * 2.0f * half_width * focus_dist;
        vertical   = v * 2.0f * half_height * focus_dist;
    }

    __device__ Ray get_ray(float s, float t, curandState *local_rand_state)
    {
        Vec3f rd     = random_in_unit_disk(local_rand_state) * lens_radius;
        Vec3f offset = u * rd.x + v * rd.y;
        return Ray(origin + offset,
                   lower_left_corner + horizontal * s + vertical * t - origin - offset);
    }

    Vec3f origin;
    Vec3f lower_left_corner;
    Vec3f horizontal;
    Vec3f vertical;
    Vec3f u, v, w;
    float lens_radius;
};

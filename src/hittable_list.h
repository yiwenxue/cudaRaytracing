#pragma once

#include "hittable.h"

#include "ray.h"

class HitTableList : public HitTable
{
public:
    __device__ HitTableList(){};
    __device__ HitTableList(HitTable **l, int s) : list(l), size(s){};

    __device__ virtual bool hit(const Ray &r, float t_min, float t_max,
                                HitRecord &rec) const override;

public:
    HitTable **list;
    int        size;
};

__device__ bool HitTableList::hit(const Ray &r, float t_min, float t_max, HitRecord &rec) const
{
    HitRecord temp_rec;
    auto      hit_anything   = false;
    auto      closest_so_far = t_max;

    for (int i = 0; i < size; i++)
    {
        if (list[i]->hit(r, t_min, closest_so_far, temp_rec))
        {
            hit_anything   = true;
            closest_so_far = temp_rec.t;
            rec            = temp_rec;
        }
    }

    return hit_anything;
}

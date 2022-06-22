#pragma once

#include "primitive.h"

#include <vector>

class Aggregate
{
public:
    Aggregate(const std::vector<Primitive *> prims, int max_depth = 5);
    ~Aggregate();

    virtual bool intersect(const Ray &ray, float hit) const = 0;

protected:
    std::vector<Primitive *> primitives{};
    int                      max_depth{5};
};

struct BVHBuildNode
{
    BVHBuildNode()
    {
        children[0] = nullptr;
        children[1] = nullptr;
    }

    CPU_GPU inline void initLeaf(int first, int n, const AABB b)
    {
        firstPrimitive = first;
        nPrimitives    = n;
        bound          = b;
    }
    CPU_GPU inline void initInterior(int axis, BVHBuildNode *c0, BVHBuildNode *c1)
    {
        children[0] = c0;
        children[1] = c1;
        bound       = Union(c0->bound, c1->bound);
        splitx      = axis;
    }

    AABB          bound;
    BVHBuildNode *children[2];

    int splitx{0};
    int firstPrimitive{0};
    int nPrimitives{0};
};
void printBVH(BVHBuildNode *node, int depth = 0);

struct BVHPrimitive;

class BVHAggregate : public Aggregate
{
public:
    enum class SplitMethod
    {
        SAH,    //
        LBVH,   //
        MIDDLE, //
    };

    BVHAggregate(const std::vector<Primitive *> prims, int max_depth = 5);
    ~BVHAggregate();

    bool intersect(const Ray &ray, float hit) const override;

    bool bvhNodeIntersect(const BVHBuildNode *node, const Ray &ray, float tMax = FLT_MAX) const;

    BVHBuildNode *buildRecursive(std::vector<BVHPrimitive> bvhPrimitives, size_t totalNodes,
                                 size_t                    orderedPrimOffset,
                                 std::vector<Primitive *> &orderedPrimitives);

protected:
    BVHBuildNode *root{nullptr};
};

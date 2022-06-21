#include "aggregates.h"
#include <algorithm>
#include <cstdio>

Aggregate::Aggregate(const std::vector<Primitive *> prims, int max_depth) :
    primitives(prims), max_depth(max_depth)
{}

Aggregate::~Aggregate()
{
    this->primitives.clear();
}

BVHAggregate::BVHAggregate(const std::vector<Primitive *> prims, int max_depth) :
    Aggregate(prims, max_depth)
{
    printf("Start to build a bvh aggregate.\n");

    std::vector<Primitive *>  orderedPrims(primitives.size());
    std::vector<BVHPrimitive> bvhPrimitives(primitives.size());

    for (size_t i = 0; i < primitives.size(); ++i)
        bvhPrimitives[i] = BVHPrimitive(i, primitives[i]->getBounds());

    BVHBuildNode *root = buildRecursive(bvhPrimitives, primitives.size(), 0, orderedPrims);

    primitives.swap(orderedPrims);
}

struct BVHPrimitive
{
    BVHPrimitive(size_t primitiveIndex, const AABB &bounds) :
        primitiveIndex(primitiveIndex), bounds(bounds)
    {}

    Vec3f getCenter() const
    {
        return bounds.aabb[0] * 0.5 + bounds.aabb[1] * 0.5;
    }

    size_t primitiveIndex;
    AABB   bounds;
};

BVHBuildNode *BVHAggregate::buildRecursive(std::vector<BVHPrimitive> bvhPrimitives,
                                           size_t totalNodes, size_t orderedPrimOffset,
                                           std::vector<Primitive *> &orderedPrimitives)
{
    AABB bounds;
    for (const auto &prim : bvhPrimitives)
        bounds = Union(bounds, prim.bounds);

    BVHBuildNode *node = new BVHBuildNode();
    totalNodes++;

    // TODO yiwenxue: should use area here, sine 2d shape also has 0 volume
    if (bounds.getVolume() == 0. || bvhPrimitives.size() == 1)
    {
        // atomic for MP
        orderedPrimOffset += bvhPrimitives.size();
        size_t firstPrimOffset = orderedPrimOffset;

        for (size_t i = 0; i < bvhPrimitives.size(); i++)
        {
            int idx                                = bvhPrimitives[i].primitiveIndex;
            orderedPrimitives[firstPrimOffset + i] = primitives[idx];
        }
        node->initLeaf(firstPrimOffset, bvhPrimitives.size(), bounds);
        return node;
    }
    else
    {
        AABB centroidBounds;
        for (const auto &prim : bvhPrimitives)
            centroidBounds = Union(centroidBounds, prim.getCenter());

        uint dim = centroidBounds.maxDimension();
        int  mid = bvhPrimitives.size() / 2;

        {
            // mid point to split bvh
            float pmid = (centroidBounds.aabb[0][dim] + centroidBounds.aabb[1][dim]) * 0.5;
            std::partition(
                bvhPrimitives.begin(), bvhPrimitives.end(),
                [dim, pmid](const BVHPrimitive &prim) { return prim.getCenter()[dim] < pmid; });
        }

        BVHBuildNode *children[2];

        std::vector<BVHPrimitive>::const_iterator split = bvhPrimitives.begin() + mid;

        std::vector<BVHPrimitive> front(bvhPrimitives.cbegin(), split);
        std::vector<BVHPrimitive> back(split, bvhPrimitives.cend());

        // recursively build all nodes
        children[0] = buildRecursive(front, totalNodes, orderedPrimOffset, orderedPrimitives);
        children[1] = buildRecursive(back, totalNodes, orderedPrimOffset, orderedPrimitives);

        node->initInterior(dim, children[0], children[1]);
    }

    return node;
}

BVHAggregate::~BVHAggregate()
{}

bool BVHAggregate::intersect(const Ray &ray, float hit) const
{
    return false;
}

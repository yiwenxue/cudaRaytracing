#include "aggregates.h"
#include <algorithm>
#include <cstdio>

struct BVHPrimitive
{
    BVHPrimitive() = default;

    BVHPrimitive(size_t primitiveIndex, const AABB bounds) :
        primitiveIndex(primitiveIndex), bounds(bounds)
    {}

    Vec3f getCenter() const
    {
        return bounds.pMin * 0.5 + bounds.pMax * 0.5;
    }

    size_t primitiveIndex{0};
    AABB   bounds{};
};

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
    std::vector<Primitive *>  orderedPrims(primitives.size());
    std::vector<BVHPrimitive> bvhPrimitives(primitives.size());

    for (size_t i = 0; i < primitives.size(); ++i)
        bvhPrimitives[i] = BVHPrimitive(i, primitives[i]->getBounds());

    BVHBuildNode *root = buildRecursive(bvhPrimitives, primitives.size(), 0, orderedPrims);

    primitives.swap(orderedPrims);

    printBVH(root);
}

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
            float pmid = (centroidBounds.pMin[dim] + centroidBounds.pMax[dim]) * 0.5;
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

bool BVHAggregate::bvhNodeIntersect(const BVHBuildNode *node, const Ray &ray, float tMax) const
{
    if (node->nPrimitives > 0)
    {
        for (size_t i = 0; i < node->firstPrimitive; i++)
        {
            if (primitives[node->firstPrimitive + i]->intersectP(ray, tMax))
                return true;
        }
        return false;
    }
    else
    {
        if (bvhNodeIntersect(node->children[0], ray, tMax))
            return true;
        if (bvhNodeIntersect(node->children[1], ray, tMax))
            return true;
        return false;
    }
}
bool BVHAggregate::intersect(const Ray &ray, float tMax) const
{
    return bvhNodeIntersect(root, ray, tMax);
}

void printBVH(BVHBuildNode *node, int depth)
{
    if (node->nPrimitives > 0)
    {
        printf("%*sLeaf %d: %d nodes\n", depth * 2, "", node->firstPrimitive, node->nPrimitives);
    }
    else
    {
        printf("%*sInterior %d: %d nodes\n", depth * 2, "", node->splitx, node->nPrimitives);
        printBVH(node->children[0], depth + 1);
        printBVH(node->children[1], depth + 1);
    }
}

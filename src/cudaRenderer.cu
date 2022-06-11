#include "camera.h"
#include "hittable_list.h"
#include "material.h"
#include "ray.h"
#include "sphere.h"
#include <curand_kernel.h>
#include <float.h>
#include <iostream>
#include <time.h>
#include <vector>

#include "cudaRenderer.h"

#include "common/CudaMath.h"
#include "common/CudaArray.h"
#include "common/CudaUtils.h"

#include "stb_image_write.h"

#define SCENE_OBJECTS 10

__device__ Vec3f color(const Ray &r, HitTable **world, curandState *local_rand_state)
{
    Ray   cur_ray         = r;
    Vec3f cur_attenuation = Vec3f(1.0, 1.0, 1.0);
    for (int i = 0; i < 200; i++)
    {
        HitRecord rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec))
        {
            Ray   scattered;
            Vec3f attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
            {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else
            {
                return Vec3f(0.0, 0.0, 0.0);
            }
        }
        else
        {
            Vec3f unit_direction = Vec3f::normal(cur_ray.direct);
            float t              = 0.5f * (unit_direction.y + 1.0f);
            Vec3f c              = Vec3f(1.0, 1.0, 1.0) * (1.0f - t) + Vec3f(0.5, 0.7, 1.0) * t;
            return cur_attenuation * c;
        }
    }
    return Vec3f(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;
    // Original: Each thread gets same seed, a different sequence number, no offset
    // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
    // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
    // performance improvement of about 2x!
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(cudaSurfaceObject_t surf, float4 * accumulate,int max_x, int max_y, int ns, Camera **cam, HitTable **world,
                       curandState *rand_state, int framecount)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    
    int         pixel_index      = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    float x = curand_uniform(&local_rand_state);
    float y = curand_uniform(&local_rand_state);
    float z = curand_uniform(&local_rand_state);

    Vec3f       col(0, 0, 0);
    for (int s = 0; s < ns; s++)
    {
        float u = float(i + x) / float(max_x);
        float v = float(j + y) / float(max_y);
        Ray   r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);

    accumulate[pixel_index].x += 255.99 * sqrt(col.x);
    accumulate[pixel_index].y += 255.99 * sqrt(col.y);
    accumulate[pixel_index].z += 255.99 * sqrt(col.z);
    accumulate[pixel_index].w += 255;

    uchar4 data;

    data.x = (accumulate[pixel_index].x / framecount);
    data.y = (accumulate[pixel_index].y / framecount);
    data.z = (accumulate[pixel_index].z / framecount);
    data.w = (accumulate[pixel_index].w / framecount);

    surf2Dwrite(data, surf, i * sizeof(uchar4), max_y - j - 1,
        cudaBoundaryModeZero
    );
}


#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(HitTable **d_list, HitTable **d_world, Camera **d_camera, int nx,
                             int ny, curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        d_list[0] = new Sphere(Vec3f(0,-500.0,-1), 500,
                        new Metal(Vec3f(0.7, 0.6, 0.5), 0.0));

        curandState local_rand_state = *rand_state;
        int         i                = 1;
        for (int a = -SCENE_OBJECTS; a < SCENE_OBJECTS; a++)
        {
            for (int b = -SCENE_OBJECTS; b < SCENE_OBJECTS; b++)
            {
                float choose_mat = RND;
                Vec3f center(a + RND, 0.2, b + RND);

                if(choose_mat < 0.3f) {
                    d_list[i++] = new Sphere(center, 0.2,
                                             new Dielectric(RND + 1.0));
                } else if(choose_mat < 0.8f) {
                    d_list[i++] = new Sphere(
                        center, 0.2,
                        new Lambertian(Vec3f(RND*RND, RND*RND, RND*RND)));
                } else{
                    d_list[i++] = new Sphere(
                        center, 0.2,
                        new Metal(Vec3f(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)),
                                0.5f * RND));
                }
            }
        }
        d_list[i++] = new Sphere(Vec3f(0, 1,0), 1.0, new Dielectric(1.5));
        d_list[i++] = new Sphere(Vec3f(-4, 1, 0), 1.0, new Lambertian(Vec3f(0.4, 0.2, 0.1)));
        d_list[i++] = new Sphere(Vec3f(4, 1, 0), 1.0, new Metal(Vec3f(0.7, 0.6, 0.7), 0.0));

        *rand_state = local_rand_state;
        *d_world    = new HitTableList(d_list, SCENE_OBJECTS * SCENE_OBJECTS * 4 + 1 + 3);

        Vec3f lookfrom(13, 2, 3);
        Vec3f lookat(0, 0, 0);
        float dist_to_focus = 10.0;
        (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_camera      = new Camera(lookfrom, lookat, Vec3f(0, 1, 0), 30.0, float(nx) / float(ny),
                                    aperture, dist_to_focus);
    }
}

__global__ void free_world(HitTable **d_list, HitTable **d_world, Camera **d_camera)
{
    for (int i = 0; i < SCENE_OBJECTS * SCENE_OBJECTS * 4 + 1 + 3; i++)
    {
        delete ((Sphere *) d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

__global__ void simple_kernel(cudaSurfaceObject_t surf, int w, int h) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= w) || (j >= h)) {
        return;
    }

    uchar4 data;

    surf2Dread(&data, surf, i*sizeof(uchar4) , j);

    data.x += 1;
    data.y += 1;
    data.z += 1;

    surf2Dwrite(data, surf,
        i * sizeof(uchar4), j,
        cudaBoundaryModeZero
    );
}

#include "renderer.h"

void Renderer::cudaRenderInit() {

    int num_pixels = _bufferWidth * _bufferHeight;
    // allocate random state
    checkCudaErrors(cudaMalloc((void **) &d_rand_state, num_pixels * sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void **) &d_rand_state2, 1 * sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1, 1>>>(static_cast<curandState *>(d_rand_state2));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    int        num_hitables = SCENE_OBJECTS * SCENE_OBJECTS * 4 + 1 + 3;
    checkCudaErrors(cudaMalloc((void **) &d_list, num_hitables * sizeof(HitTable *)));
    checkCudaErrors(cudaMalloc((void **) &d_world, sizeof(HitTable *)));
    checkCudaErrors(cudaMalloc((void **) &d_camera, sizeof(Camera *)));

    create_world<<<1, 1>>>(d_list, d_world, d_camera, _bufferWidth, _bufferHeight, static_cast<curandState *>(d_rand_state2));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    int tx = 16;
    int ty = 16;

    dim3 blocks( _bufferWidth/ tx + 1, _bufferHeight / ty + 1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(_bufferWidth, _bufferHeight, static_cast<curandState *>(d_rand_state));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void Renderer::cudaRenderDraw() {
    int tx = 16;
    int ty = 16;
    int ns = 1;

    dim3 blocks(_bufferWidth / tx + 1, _bufferHeight / ty + 1);
    dim3 threads(tx, ty);
    float duration = clock();
    render<<<blocks, threads>>>(cudaSurf, _cuda_res, _bufferWidth, _bufferHeight, ns, d_camera, d_world, static_cast<curandState *>(d_rand_state), framecount);
    // simple_kernel<<<blocks, threads>>>(cudaSurf, _bufferWidth, _bufferHeight);
    cudaDeviceSynchronize();
    duration = clock() - duration;
    duration /= CLOCKS_PER_SEC / 1000.0;

    int idx = framecount % 60;
    total_time += duration - frame_time[idx];
    frame_time[idx] = duration;
}

void Renderer::cudaRenderClean() {
    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));

    cudaDeviceReset();
}

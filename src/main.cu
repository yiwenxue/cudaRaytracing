#include "camera.h"
#include "hittable_list.h"
#include "material.h"
#include "ray.h"
#include "sphere.h"
#include <curand_kernel.h>
#include <float.h>
#include <iostream>
#include <time.h>

#include "main.h"

#include "common/CudaMath.h"
#include "common/CudaArray.h"
#include "common/CudaUtils.h"

#include "stb_image_write.h"

__device__ Vec3f color(const Ray &r, HitTable **world, curandState *local_rand_state)
{
    Ray   cur_ray         = r;
    Vec3f cur_attenuation = Vec3f(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++)
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

__global__ void render(Vec3f *fb, int max_x, int max_y, int ns, Camera **cam, HitTable **world,
                       curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int         pixel_index      = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    Vec3f       col(0, 0, 0);
    for (int s = 0; s < ns; s++)
    {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        Ray   r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col.x           = sqrt(col.x);
    col.y           = sqrt(col.y);
    col.z           = sqrt(col.z);
    fb[pixel_index] = col;
}


#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(HitTable **d_list, HitTable **d_world, Camera **d_camera, int nx,
                             int ny, curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        d_list[0] = new Sphere(Vec3f(0,-1000.0,-1), 1000,
                        new Metal(Vec3f(0.7, 0.6, 0.5), 0.0));

        curandState local_rand_state = *rand_state;
        int         i                = 1;
        for (int a = -11; a < 11; a++)
        {
            for (int b = -11; b < 11; b++)
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
        *d_world    = new HitTableList(d_list, 22 * 22 + 1 + 3);

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
    for (int i = 0; i < 22 * 22 + 1 + 3; i++)
    {
        delete ((Sphere *) d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

void ray_tracing()
{
    int nx = 1200;
    int ny = 800;
    int ns = 20;

    int tx = 16;
    int ty = 16;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int    num_pixels = nx * ny;
    size_t fb_size    = num_pixels * sizeof(Vec3f);

    // allocate FB
    Vec3f *fb;
    checkCudaErrors(cudaMallocManaged((void **) &fb, fb_size));

    // allocate random state
    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **) &d_rand_state, num_pixels * sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **) &d_rand_state2, 1 * sizeof(curandState)));

    // we need that 2nd random state to be initialized for the world creation
    rand_init<<<1, 1>>>(d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hitables & the camera
    HitTable **d_list;
    int        num_hitables = 22 * 22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void **) &d_list, num_hitables * sizeof(HitTable *)));
    HitTable **d_world;
    checkCudaErrors(cudaMalloc((void **) &d_world, sizeof(HitTable *)));
    Camera **d_camera;
    checkCudaErrors(cudaMalloc((void **) &d_camera, sizeof(Camera *)));
    create_world<<<1, 1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start, stop;
    start = clock();
    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    stop                 = clock();
    double timer_seconds = ((double) (stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    // Output FB as Image
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--)
    {
        for (int i = 0; i < nx; i++)
        {
            size_t pixel_index = j * nx + i;
            int    ir          = int(255.99 * fb[pixel_index].x);
            int    ig          = int(255.99 * fb[pixel_index].y);
            int    ib          = int(255.99 * fb[pixel_index].z);
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1>>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(fb));

    cudaDeviceReset();
}

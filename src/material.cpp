// #include "material.h"

// #define RANDVEC3                                                              \
//     Vec3f(curand_uniform(local_rand_state), curand_uniform(local_rand_state), \
//           curand_uniform(local_rand_state))

// __device__ Vec3f random_in_unit_sphere(curandState *local_rand_state)
// {
//     Vec3f p;
//     do
//     {
//         p = RANDVEC3 * 2.0 - Vec3f(1, 1, 1);
//     } while (p.lengthsq() >= 1.0f);
//     return p;
// }

// __device__ Vec3f refract(const Vec3f &v, const Vec3f &n, float ni_over_nt)
// {
//     auto  cos_theta      = MIN(-Vec3f::dot(v, n), 1.0);
//     Vec3f r_out_perp     = (v + n * cos_theta) * ni_over_nt;
//     Vec3f r_out_parallel = n * -sqrt(fabs(1.0 - r_out_perp.lengthsq()));
//     return r_out_perp + r_out_parallel;
// }

// __device__ float reflectance(float cosine, float ref_idx)
// {
//     // Use Schlick's approximation for reflectance.
//     float r0 = (1 - ref_idx) / (1 + ref_idx);
//     r0       = r0 * r0;
//     return r0 + (1 - r0) * pow((1 - cosine), 5);
// }

// __device__ bool Lambertian::scatter(const Ray &r_in, const HitRecord &rec, Vec3f &attenuation,
//                                     Ray &scattered, curandState *local_rand_state) const
// {
//     Vec3f target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
//     scattered    = Ray(rec.p, target - rec.p);
//     attenuation  = albedo;
//     return true;
// }

// __device__ bool Metal::scatter(const Ray &r_in, const HitRecord &rec, Vec3f &attenuation,
//                                Ray &scattered, curandState *local_rand_state) const
// {
//     Vec3f reflected = reflect(Vec3f::normal(r_in.direct), rec.normal);
//     scattered       = Ray(rec.p, reflected + random_in_unit_sphere(local_rand_state) * fuzz);
//     attenuation     = albedo;
//     return (dot(scattered.direct, rec.normal) > 0);
// }

// __device__ bool Dielectric::scatter(const Ray &r_in, const HitRecord &rec, Vec3f &attenuation,
//                                     Ray &scattered, curandState *local_rand_state) const
// {
//     attenuation            = Vec3f(1.0, 1.0, 1.0);
//     float refraction_ratio = rec.front_face ? (1.0 / ref_idx) : ref_idx;

//     Vec3f unit_direction = Vec3f::normal(r_in.direct);
//     float cos_theta      = MIN(dot(unit_direction * -1, rec.normal), 1.0);
//     float sin_theta      = sqrt(1.0 - cos_theta * cos_theta);

//     bool  cannot_refract = refraction_ratio * sin_theta > 1.0;
//     Vec3f direction;

//     if (cannot_refract || reflectance(cos_theta, refraction_ratio) > randomf())
//         direction = reflect(unit_direction, rec.normal);
//     else
//         direction = refract(unit_direction, rec.normal, refraction_ratio);

//     scattered = Ray(rec.p, direction);

//     return true;
// }

#pragma once

#include <float.h>

#include <cuda_runtime.h>
////////////////////////////////////////////////////////////////////////////////
typedef unsigned int   uint;
typedef unsigned short ushort;

#define INF 2e10f

#ifndef __CUDACC__
#    include <math.h>

inline float fminf(float a, float b)
{
    return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
    return a > b ? a : b;
}

inline int max(int a, int b)
{
    return a > b ? a : b;
}

inline int min(int a, int b)
{
    return a < b ? a : b;
}

inline float rsqrtf(float x)
{
    return 1.0f / sqrtf(x);
}
#endif

// float functions
////////////////////////////////////////////////////////////////////////////////

// lerp
inline __device__ __host__ float lerp(float a, float b, float t)
{
    return a + t * (b - a);
}

// clamp
inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}

// float3 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
inline __host__ __device__ float3 make_float3(float s)
{
    return make_float3(s, s, s);
}
inline __host__ __device__ float3 make_float3(float2 a)
{
    return make_float3(a.x, a.y, 0.0f);
}
inline __host__ __device__ float3 make_float3(float2 a, float s)
{
    return make_float3(a.x, a.y, s);
}
inline __host__ __device__ float3 make_float3(float4 a)
{
    return make_float3(a.x, a.y, a.z); // discards w
}
inline __host__ __device__ float3 make_float3(int3 a)
{
    return make_float3(float(a.x), float(a.y), float(a.z));
}

// negate
inline __host__ __device__ float3 operator-(float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

// min
static __inline__ __host__ __device__ float3 fminf(float3 a, float3 b)
{
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

// max
static __inline__ __host__ __device__ float3 fmaxf(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

// addition
inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

// subtract
inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ float3 operator-(float3 a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}
inline __host__ __device__ void operator-=(float3 &a, float3 b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

// multiply
inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ float3 operator*(float3 a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ float3 operator*(float s, float3 a)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
inline __host__ __device__ void operator*=(float3 &a, float s)
{
    a.x *= s;
    a.y *= s;
    a.z *= s;
}
inline __host__ __device__ void operator*=(float3 &a, float3 b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
}

// divide
inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ float3 operator/(float3 a, float s)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ float3 operator/(float s, float3 a)
{
    float inv = 1.0f / s;
    return a * inv;
}
inline __host__ __device__ void operator/=(float3 &a, float s)
{
    float inv = 1.0f / s;
    a *= inv;
}

// lerp
inline __device__ __host__ float3 lerp(float3 a, float3 b, float t)
{
    return a + t * (b - a);
}

// clamp
inline __device__ __host__ float3 clamp(float3 v, float a, float b)
{
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline __device__ __host__ float3 clamp(float3 v, float3 a, float3 b)
{
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

// dot product
inline __host__ __device__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// cross product
inline __host__ __device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// length
inline __host__ __device__ float length(float3 v)
{
    return sqrtf(dot(v, v));
}

inline __host__ __device__ float lengthSq(float3 v)
{
    return dot(v, v);
}

// normalize
inline __host__ __device__ float3 normalize(float3 v)
{
    float invLen = rsqrtf(dot(v, v));
    return v * invLen;
}

// floor
inline __host__ __device__ float3 floor(const float3 v)
{
    return make_float3(floor(v.x), floor(v.y), floor(v.z));
}

// reflect
inline __host__ __device__ float3 reflect(float3 i, float3 n)
{
    return i - 2.0f * n * dot(n, i);
}

// absolute value
inline __host__ __device__ float3 fabs(float3 v)
{
    return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}

typedef unsigned int U32;
typedef float        F32;

#define rnd(x) (x * rand() / RAND_MAX)

inline __host__ __device__ float randomf()
{
    return rand() / (RAND_MAX + 1.0);
}

inline __host__ __device__ float randomf(float min, float max)
{
    return min + (max - min) * randomf();
}

struct Vec2f
{
    /// float x, y;
    union
    {
        struct
        {
            float x, y;
        };
        float _v[2];
    };

    __host__ __device__ Vec2f(float _x = 0, float _y = 0) : x(_x), y(_y)
    {}
    __host__ __device__ Vec2f(const Vec2f &v) : x(v.x), y(v.y)
    {}

    inline __host__ __device__ bool operator==(const Vec2f &v)
    {
        return x == v.x && y == v.y;
    }
    inline __host__ __device__ Vec2f operator+(const Vec2f &v) const
    {
        return Vec2f(x + v.x, y + v.y);
    }
    inline __host__ __device__ Vec2f operator-(const Vec2f &v) const
    {
        return Vec2f(x - v.x, y - v.y);
    }
    inline __host__ __device__ Vec2f operator*(const Vec2f &v) const
    {
        return Vec2f(x * v.x, y * v.y);
    }
};

struct Vec2i
{
    /// int x, y;
    union
    {
        struct
        {
            int x, y;
        };
        int _v[2];
    };

    __host__ __device__ Vec2i(int _x = 0, int _y = 0) : x(_x), y(_y)
    {}
    __host__ __device__ Vec2i(const Vec2i &v) : x(v.x), y(v.y)
    {}

    inline bool __host__ __device__ operator==(const Vec2i &v)
    {
        return x == v.x && y == v.y;
    }
};

inline __host__ __device__ float max1f(const float &a, const float &b)
{
    return (a < b) ? b : a;
}
inline __host__ __device__ float min1f(const float &a, const float &b)
{
    return (a > b) ? b : a;
}

struct Vec3f
{
    // float x, y, z;
    union
    {
        struct
        {
            float x, y, z;
        };
        float _v[3];
    };

    __host__ __device__ Vec3f(float _x = 0, float _y = 0, float _z = 0) : x(_x), y(_y), z(_z)
    {}
    __host__ __device__ Vec3f(const Vec3f &v) : x(v.x), y(v.y), z(v.z)
    {}
    inline __host__ __device__ float length() const
    {
        return sqrtf(x * x + y * y + z * z);
    }
    // sometimes we dont need the sqrt, we are just comparing one length with another
    inline __host__ __device__ float lengthsq() const
    {
        return x * x + y * y + z * z;
    }
    inline __host__ __device__ float max() const
    {
        return max1f(max1f(x, y), z);
    }
    inline __host__ __device__ float min() const
    {
        return min1f(min1f(x, y), z);
    }
    inline __host__ __device__ Vec3f normalize()
    {
        float norm = sqrtf(x * x + y * y + z * z);
        x /= norm;
        y /= norm;
        z /= norm;
        return Vec3f(x, y, z);
    }
    inline __host__ __device__ static Vec3f random(double min, double max)
    {
        return Vec3f(randomf(min, max), randomf(min, max), randomf(min, max));
    }

    inline __host__ __device__ static float dot(Vec3f a, Vec3f b)
    {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    inline __host__ __device__ static Vec3f normal(const Vec3f &v)
    {
        return v / v.length();
    }

    inline __host__ __device__ static Vec3f cross(Vec3f a, Vec3f b)
    {
        return Vec3f(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    }

    inline __host__ __device__ float &operator[](const int idx)
    {
        return _v[idx];
    }

    inline __host__ __device__ const float &operator[](const int idx) const
    {
        return _v[idx];
    }

    inline __host__ __device__ Vec3f &operator+=(const Vec3f &v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }
    inline __host__ __device__ Vec3f &operator-=(const Vec3f &v)
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }
    inline __host__ __device__ Vec3f &operator*=(const float &a)
    {
        x *= a;
        y *= a;
        z *= a;
        return *this;
    }
    inline __host__ __device__ Vec3f &operator*=(const Vec3f &v)
    {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }
    inline __host__ __device__ Vec3f operator*(float a) const
    {
        return Vec3f(x * a, y * a, z * a);
    }
    inline __host__ __device__ Vec3f operator/(float a) const
    {
        return Vec3f(x / a, y / a, z / a);
    }
    inline __host__ __device__ Vec3f operator*(const Vec3f &v) const
    {
        return Vec3f(x * v.x, y * v.y, z * v.z);
    }
    inline __host__ __device__ Vec3f operator+(const Vec3f &v) const
    {
        return Vec3f(x + v.x, y + v.y, z + v.z);
    }
    inline __host__ __device__ Vec3f operator-(const Vec3f &v) const
    {
        return Vec3f(x - v.x, y - v.y, z - v.z);
    }
    inline __host__ __device__ Vec3f operator/(const Vec3f &v) const
    {
        return Vec3f(x / v.x, y / v.y, z / v.z);
    }
    inline __host__ __device__ Vec3f &operator/=(const float &a)
    {
        x /= a;
        y /= a;
        z /= a;
        return *this;
    }
    inline __host__ __device__ bool operator!=(const Vec3f &v)
    {
        return x != v.x || y != v.y || z != v.z;
    }
    inline __host__ __device__ bool operator==(const Vec3f &v)
    {
        return x == v.x && y == v.y && z == v.z;
    }
};

inline __host__ __device__ Vec3f reflect(const Vec3f &v, const Vec3f &n)
{
    return v - n * 2.0 * Vec3f::dot(v, n);
}

struct Vec3i
{
    union
    {
        struct
        {
            int x, y, z;
        };
        int _v[3];
    };
    /// int x, y, z;

    __host__ __device__ Vec3i(int _x = 0, int _y = 0, int _z = 0) : x(_x), y(_y), z(_z)
    {}
    __host__ __device__ Vec3i(const Vec3i &v) : x(v.x), y(v.y), z(v.z)
    {}
    __host__ __device__ Vec3i(const Vec3f &vf) : x(vf.x), y(vf.y), z(vf.z)
    {}

    inline __host__ __device__ bool operator==(const Vec3i &v)
    {
        return x == v.x && y == v.y && z == v.z;
    }
};

struct Vec4f
{
    union
    {
        struct
        {
            float x, y, z, w;
        };
        float _v[4];
    };
    /// float x, y, z, w;

    __host__ __device__ Vec4f(float _x = 0, float _y = 0, float _z = 0, float _w = 0) :
        x(_x), y(_y), z(_z), w(_w)
    {}
    __host__ __device__ Vec4f(const Vec4f &v) : x(v.x), y(v.y), z(v.z), w(v.w)
    {}
    __host__ __device__ Vec4f(const Vec3f &v, const float a) : x(v.x), y(v.y), z(v.z), w(a)
    {}

    inline __host__ __device__ Vec4f &operator+=(const Vec4f &v)
    {
        x += v.x;
        y += v.y;
        z += v.z;
        w += v.w;
        return *this;
    }
    inline __host__ __device__ Vec4f &operator*=(const Vec4f &v)
    {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        w *= v.w;
        return *this;
    }
};

struct Vec4i
{
    int x, y, z, w;

    __host__ __device__ Vec4i(int _x = 0, int _y = 0, int _z = 0, int _w = 0) :
        x(_x), y(_y), z(_z), w(_w)
    {}
    __host__ __device__ Vec4i(const Vec4i &v) : x(v.x), y(v.y), z(v.z), w(v.w)
    {}
    __host__ __device__ Vec4i(const Vec3i &v, const int a) : x(v.x), y(v.y), z(v.z), w(a)
    {}
};

inline __host__ __device__ Vec3f min3f(const Vec3f &v1, const Vec3f &v2)
{
    return Vec3f(v1.x < v2.x ? v1.x : v2.x, v1.y < v2.y ? v1.y : v2.y, v1.z < v2.z ? v1.z : v2.z);
}
inline __host__ __device__ Vec3f max3f(const Vec3f &v1, const Vec3f &v2)
{
    return Vec3f(v1.x > v2.x ? v1.x : v2.x, v1.y > v2.y ? v1.y : v2.y, v1.z > v2.z ? v1.z : v2.z);
}
inline __host__ __device__ Vec3f cross(const Vec3f &v1, const Vec3f &v2)
{
    return Vec3f(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}
inline __host__ __device__ float dot(const Vec3f &v1, const Vec3f &v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
inline __host__ __device__ float dot(const Vec4f &v1, const Vec4f &v2)
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
inline __host__ __device__ float distancesq(const Vec3f &v1, const Vec3f &v2)
{
    return (v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y) +
           (v1.z - v2.z) * (v1.z - v2.z);
}
inline __host__ __device__ float distance(const Vec3f &v1, const Vec3f &v2)
{
    return sqrtf((v1.x - v2.x) * (v1.x - v2.x) + (v1.y - v2.y) * (v1.y - v2.y) +
                 (v1.z - v2.z) * (v1.z - v2.z));
}
inline __host__ __device__ Vec3f powf(const Vec3f &v1, const Vec3f &v2)
{
    return Vec3f(powf(v1.x, v2.x), powf(v1.y, v2.y), powf(v1.z, v2.z));
}
inline __host__ __device__ Vec3f expf(const Vec3f &v)
{
    return Vec3f(expf(v.x), expf(v.y), expf(v.z));
}
inline __host__ __device__ float clampf(float a, float lo, float hi)
{
    return a < lo ? lo : a > hi ? hi : a;
}
inline __host__ __device__ Vec3f mixf(const Vec3f &v1, const Vec3f &v2, float a)
{
    return v1 * (1.0 - a) + v2 * a;
}
inline __host__ __device__ float smoothstep(float edge0, float edge1, float x)
{
    float t;
    t = clampf((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

//-------------------------------------------------------------------------------------------------

//----------------------------------
// Matrix algebra
//----------------------------------

// based on Aila Laine framework , without templates

class Mat3f
{
public:
    inline void set(const float *ptr)
    {
        for (int i = 0; i < 3 * 3; i++)
            get(i) = ptr[i];
    }
    inline void set(const float &a)
    {
        for (int i = 0; i < 3 * 3; i++)
            get(i) = a;
    }
    inline void setZero(void)
    {
        set((float) 0);
    }
    inline void setIdentity(void)
    {
        setZero();
        for (int i = 0; i < 3; i++)
            get(i, i) = (float) 1;
    }
    inline const float &get(int idx) const
    {
        return getPtr()[idx];
    }
    inline float &get(int idx)
    {
        return getPtr()[idx];
    }
    inline const float &get(int r, int c) const
    {
        return getPtr()[r + c * 3];
    }
    inline float &get(int r, int c)
    {
        return getPtr()[r + c * 3];
    }
    inline void set(const Mat3f &v)
    {
        set(v.getPtr());
    }
    inline Mat3f &operator=(const float &a)
    {
        set(a);
        return *(Mat3f *) this;
    }
    inline const float &operator()(int r, int c) const
    {
        return get(r, c);
    }
    inline float &operator()(int r, int c)
    {
        return get(r, c);
    }
    inline float det(void) const;

    inline Mat3f(void)
    {
        setIdentity();
    }
    inline explicit Mat3f(F32 a)
    {
        set(a);
    }

    inline const F32 *getPtr(void) const
    {
        return &m00;
    }
    inline F32 *getPtr(void)
    {
        return &m00;
    }
    static inline Mat3f fromPtr(const F32 *ptr)
    {
        Mat3f v;
        v.set(ptr);
        return v;
    }

    inline Mat3f(const Mat3f &v)
    {
        set(v);
    }
    inline Mat3f &operator=(const Mat3f &v)
    {
        set(v);
        return *this;
    }

#if !FW_CUDA
    static Mat3f
        rotation(const Vec3f &axis,
                 F32 angle); // Rotation of "angle" radians around "axis". Axis must be unit!
#endif

public:
    F32 m00, m10, m20;
    F32 m01, m11, m21;
    F32 m02, m12, m22;
};

class Mat4f
{
public:
    inline const F32 *getPtr(void) const
    {
        return &m00;
    }
    inline F32 *getPtr(void)
    {
        return &m00;
    }
    inline const Vec4f &col(int c) const
    {
        return *(const Vec4f *) (getPtr() + c * 4);
    }
    inline Vec4f &col(int c)
    {
        return *(Vec4f *) (getPtr() + c * 4);
    }
    inline const Vec4f &getCol(int c) const
    {
        return col(c);
    }
    inline Vec4f getCol0() const
    {
        Vec4f col;
        col.x = m00;
        col.y = m01;
        col.z = m02;
        col.w = m03;
        return col;
    }
    inline Vec4f getCol1() const
    {
        Vec4f col;
        col.x = m10;
        col.y = m11;
        col.z = m12;
        col.w = m13;
        return col;
    }
    inline Vec4f getCol2() const
    {
        Vec4f col;
        col.x = m20;
        col.y = m21;
        col.z = m22;
        col.w = m23;
        return col;
    }
    inline Vec4f getCol3() const
    {
        Vec4f col;
        col.x = m30;
        col.y = m31;
        col.z = m32;
        col.w = m33;
        return col;
    }
    inline Vec4f getRow(int r) const;
    inline Mat4f inverted4x4(void);
    inline void  invert(void)
    {
        set(inverted4x4());
    }
    inline const float &get(int idx) const
    {
        return getPtr()[idx];
    }
    inline float &get(int idx)
    {
        return getPtr()[idx];
    }
    inline const float &get(int r, int c) const
    {
        return getPtr()[r + c * 4];
    }
    inline float &get(int r, int c)
    {
        return getPtr()[r + c * 4];
    }
    inline void set(const float &a)
    {
        for (int i = 0; i < 4 * 4; i++)
            get(i) = a;
    }
    inline void set(const float *ptr)
    {
        for (int i = 0; i < 4 * 4; i++)
            get(i) = ptr[i];
    }
    inline void setZero(void)
    {
        set((float) 0);
    }
    inline void setIdentity(void)
    {
        setZero();
        for (int i = 0; i < 4; i++)
            get(i, i) = (float) 1;
    }
    inline void setCol(int c, const Vec4f &v)
    {
        col(c) = v;
    }
    inline void setCol0(const Vec4f &v)
    {
        m00 = v.x;
        m01 = v.y;
        m02 = v.z;
        m03 = v.w;
    }
    inline void setCol1(const Vec4f &v)
    {
        m10 = v.x;
        m11 = v.y;
        m12 = v.z;
        m13 = v.w;
    }
    inline void setCol2(const Vec4f &v)
    {
        m20 = v.x;
        m21 = v.y;
        m22 = v.z;
        m23 = v.w;
    }
    inline void setCol3(const Vec4f &v)
    {
        m30 = v.x;
        m31 = v.y;
        m32 = v.z;
        m33 = v.w;
    }
    inline void setRow(int r, const Vec4f &v);
    inline void set(const Mat4f &v)
    {
        set(v.getPtr());
    }
    inline Mat4f &operator=(const float &a)
    {
        set(a);
        return *(Mat4f *) this;
    }
    inline Mat4f operator*(const float &a) const
    {
        Mat4f r;
        for (int i = 0; i < 4 * 4; i++)
            r.get(i) = get(i) * a;
        return r;
    }
    inline const float &operator()(int r, int c) const
    {
        return get(r, c);
    }
    inline float &operator()(int r, int c)
    {
        return get(r, c);
    }

    inline Mat4f(void)
    {
        setIdentity();
    }
    inline explicit Mat4f(F32 a)
    {
        set(a);
    }
    static inline Mat4f fromPtr(const F32 *ptr)
    {
        Mat4f v;
        v.set(ptr);
        return v;
    }

    inline Mat4f(const Mat4f &v)
    {
        set(v);
    }
    inline Mat4f &operator=(const Mat4f &v)
    {
        set(v);
        return *this;
    }

public:
    F32 m00, m10, m20, m30;
    F32 m01, m11, m21, m31;
    F32 m02, m12, m22, m32;
    F32 m03, m13, m23, m33;
};

inline Mat4f invert(Mat4f &v)
{
    return v.inverted4x4();
}

static const int c_popc8LUT[] = // CUDA CONST
    {
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3,
        4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4,
        4, 5, 4, 5, 5, 6, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4,
        5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5,
        4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2,
        3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
        5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4,
        5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6,
        4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
};

inline int popc8(U32 mask)
{
    return c_popc8LUT[mask & 0xFFu];
}

inline int popc16(U32 mask)
{
    return c_popc8LUT[mask & 0xFFu] + c_popc8LUT[(mask >> 8) & 0xFFu];
}

inline int popc32(U32 mask)
{
    int result = c_popc8LUT[mask & 0xFFu];
    result += c_popc8LUT[(mask >> 8) & 0xFFu];
    result += c_popc8LUT[(mask >> 16) & 0xFFu];
    result += c_popc8LUT[mask >> 24];
    return result;
}

//------------------------------------------------------------------------

Vec4f Mat4f::getRow(int idx) const
{
    Vec4f r;
    for (int i = 0; i < 4; i++)
        r._v[i] = get(idx, i);
    return r;
}

void Mat4f::setRow(int idx, const Vec4f &v)
{
    for (int i = 0; i < 4; i++)
        get(idx, i) = v._v[i];
}

inline float detImpl(const Mat3f &v)
{
    return v(0, 0) * v(1, 1) * v(2, 2) - v(0, 0) * v(1, 2) * v(2, 1) + v(1, 0) * v(2, 1) * v(0, 2) -
           v(1, 0) * v(2, 2) * v(0, 1) + v(2, 0) * v(0, 1) * v(1, 2) - v(2, 0) * v(0, 2) * v(1, 1);
}

float Mat3f::det(void) const
{
    return detImpl(*this);
}

// efficient column major matrix inversion function from http://rodolphe-vaillant.fr/?e=7

Mat4f Mat4f::inverted4x4(void)
{
    float inv[16];
    float m[16] = {m00, m10, m20, m30, m01, m11, m21, m31, m02, m12, m22, m32, m03, m13, m23, m33};

    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] +
             m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] -
             m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] +
             m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] -
              m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] -
             m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] +
             m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] -
             m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] +
              m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] +
             m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] -
             m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] +
              m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] -
              m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] -
             m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] +
             m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] -
              m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] +
              m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

    float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det == 0)
        return Mat4f();

    det = 1.f / det;
    Mat4f inverse;
    for (int i = 0; i < 16; i++)
        inverse.get(i) = inv[i] * det;

    return inverse;
}

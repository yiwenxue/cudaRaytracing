#pragma once

#include "common/CudaArray.h"
#include "common/CudaMath.h"

namespace cuda_render
{
struct CameraData
{
    Vec2f res{0, 0};
    Vec3f pos{0, 0, 0};
    Vec3f view{0, 0, 0};
    Vec3f up{0, 1, 0};
    Vec2f fov{0.5, 0.5};
    float apertureRadius{0.5};
    float focalDistance{0.5f};
};

class Camera
{
public:
    Camera();
    virtual ~Camera();

    void buildRenderCamera(CameraData *renderCamera);

    void changeYaw(float m);
    void changePitch(float m);
    void changeRadius(float m);
    void changeAltitude(float m);
    void changeFocalDistance(float m);

    void changeAperture(float m);

    void setRes(float x, float y);
    void setFovX(float x);

    void move(float m);
    void strafe(float m);
    void rotate(){}; // not implemented

private:
    void fixYaw();
    void fixPitch();
    void fixAperture();
    void fixRadius();
    void fixFocalDistance();

    Vec2f resolution{0, 0};
    Vec2f fov{0, 0};

    Vec3f centerPosition{0, 0, 0};
    Vec3f viewDirection{0, 0, -1};
    float yaw{0};
    float pitch{0};
    float radius{0};
    float apertureRadius{0};
    float focalDistance{0};
};

} // namespace cuda_render

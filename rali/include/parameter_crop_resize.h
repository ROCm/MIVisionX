#pragma once
#include <VX/vx_types.h>
#include "parameter_factory.h"

class RandomCropResizeParam
{

// +-----------------------------------------> X direction
// |  ___________________________________
// |  |   p1(x,y)      |                |
// |  |    +-----------|-----------+    |
// |  |    |           |           |    |
// |  -----------------o-----------------
// |  |    |           |           |    |
// |  |    +-----------|-----------+    |
// |  |                |        p2(x,y) |
// |  +++++++++++++++++++++++++++++++++++
// |
// V Y directoin
public:
    RandomCropResizeParam(size_t in_width_, size_t in_height_):
            in_width(in_width_),
            in_height(in_height_),
            x1(0), x2(in_width_ - 1),
            y1(0), y2(in_height_ - 1),
            x1_vx_scalar(nullptr),
            x2_vx_scalar(nullptr),
            y1_vx_scalar(nullptr),
            y2_vx_scalar(nullptr)
    {
        area_coeff = default_area();
        x_center_drift = default_x_drift();
        y_center_drift = default_y_drift();
        calculate_area(area_coeff->default_value(), x_center_drift->default_value(), y_center_drift->default_value());
    }
    void set_area_coeff(Parameter<float>* area);
    void set_x_drift(Parameter<float>* x_drift);
    void set_y_drift(Parameter<float>* y_drift);
    void create_scalars(vx_node node);
    const size_t in_width, in_height;
    size_t x1,x2, y1, y2;
    void update();
private:
    constexpr static float CROP_AREA_RANGE [2] = {0.35, 0.9};
    constexpr static float CROP_AREA_X_DRIFT_RANGE [2] = {-1.0, 1.0};
    constexpr static float CROP_AREA_Y_DRIFT_RANGE [2] = {-1.0, 1.0};
    constexpr static unsigned RESIZE_CROP_X1_OVX_PARAM_IDX = 4;
    constexpr static unsigned RESIZE_CROP_Y1_OVX_PARAM_IDX = 5;
    constexpr static unsigned RESIZE_CROP_X2_OVX_PARAM_IDX = 6;
    constexpr static unsigned RESIZE_CROP_Y2_OVX_PARAM_IDX = 7;
    vx_scalar x1_vx_scalar, x2_vx_scalar, y1_vx_scalar, y2_vx_scalar;
    constexpr static float MIN_RANDOM_AREA_COEFF = 0.05;
    Parameter<float> *area_coeff, *x_center_drift, *y_center_drift;
    void calculate_area(float area_coeff_, float x_center_drift_, float y_center_drift_);
    Parameter<float>* default_area();
    Parameter<float>* default_x_drift();
    Parameter<float>* default_y_drift();

};
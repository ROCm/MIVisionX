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
    RandomCropResizeParam(unsigned batch_size):batch_size(batch_size)
    {
        area_coeff = default_area();
        aspect_ratio_coeff = default_aspect_ratio();
        x_center_drift = default_x_drift();
        y_center_drift = default_y_drift();
        //calculate_area(area_coeff->default_value(), x_center_drift->default_value(), y_center_drift->default_value());
    }
    void set_image_dimensions(unsigned  image_idx, size_t in_width_, size_t in_height_)
    {
        in_width[image_idx] = in_width_;
        in_height[image_idx] = in_height_;
        x1[image_idx] = 0;
        x2[image_idx] = in_width_ - 1;
        y1[image_idx] = 0;
        y2[image_idx] = in_height_ - 1;
    }
    void set_area_coeff(Parameter<float>* area);
    void set_x_drift(Parameter<float>* x_drift);
    void set_y_drift(Parameter<float>* y_drift);
    void set_aspect_ratio_coeff(Parameter<float>* aspect_ratio);
    std::vector<size_t> in_width, in_height;
    std::vector<size_t> x1, x2, y1, y2;
    const unsigned batch_size;
    vx_array x1_arr, x2_arr,y1_arr, y2_arr;
    void create_array(std::shared_ptr<Graph> graph);
    void update_array();
    void update_array_for_cmn();
private:
    constexpr static float CROP_AREA_RANGE [2] = {0.05, 0.9};
    constexpr static float CROP_ASPECT_RATIO[2] = {0.7500, 1.333};
    constexpr static float CROP_AREA_X_DRIFT_RANGE [2] = {-1.0, 1.0};
    constexpr static float CROP_AREA_Y_DRIFT_RANGE [2] = {-1.0, 1.0};
    constexpr static float MIN_RANDOM_AREA_COEFF = 0.05;
    Parameter<float> *area_coeff, *x_center_drift, *y_center_drift;
    Parameter<float> *aspect_ratio_coeff;
    std::vector<size_t> x1_arr_val,x2_arr_val,y1_arr_val,y2_arr_val;
    void calculate_area_cmn(unsigned image_idx, float area_coeff_, float x_center_drift_, float y_center_drift_, float aspect_ratio_);
    Parameter<float>* default_area();
    Parameter<float>* default_aspect_ratio();
    Parameter<float>* default_x_drift();
    Parameter<float>* default_y_drift();

};
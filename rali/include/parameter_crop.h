#pragma once
#include <VX/vx_types.h>
#include "parameter_factory.h"

class CropParam
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
    CropParam() = delete;
    CropParam(unsigned int batch_size): batch_size(batch_size)
    {
        x_center_drift = default_x_drift();
        y_center_drift = default_y_drift();
    }
    void set_image_dimensions(unsigned  image_idx, size_t in_width_, size_t in_height_)
    {
        in_width[image_idx]  = in_width_;
        in_height[image_idx] = in_height_;
        // x1 = 100; //( in_width[image_idx] / 2) - (in_width[image_idx] / 4);
        // crop_w = 250;//in_width[image_idx] /2; // in_width_ - 1;
        // y1 = 100; //( in_height[image_idx] / 2) - (in_height[image_idx] / 4);
        // crop_h = 200; //in_height[image_idx] / 2;//in_height_ - 1;
        //crop_d = batch_size;
    }

    void set_x_drift(Parameter<float>* x_drift);
    void set_y_drift(Parameter<float>* y_drift);
    void set_crop_height_factor(Parameter<float>* crop_h_factor);
    void set_crop_width_factor(Parameter<float>* crop_w_factor);
    std::vector<int> in_width, in_height;
    unsigned int  x1, y1; // Should be made const
    unsigned int  crop_w, crop_h, crop_d; // Should be made Const in future
    bool centric;
    const unsigned int batch_size;
    void set_batch_size(unsigned int batch_size);
    vx_array x1_arr, y1_arr, croph_arr, cropw_arr;
    void create_array(std::shared_ptr<Graph> graph);
    void update_array();
    void fill_values(unsigned int img_idx);
private:
    constexpr static float CROP_X_DRIFT_RANGE [2]  = {0.0, 1.0}; // Normalized Co-ordinate Drift
    constexpr static float CROP_Y_DRIFT_RANGE [2]  = {0.0, 1.0};
    constexpr static float CROP_HEIGHT_FACTOR_RANGE[2]  = {0.0, 1.0}; // Normalized Crop-height
    constexpr static float CROP_WIDTH_FACTOR_RANGE[2]   = {0.0, 1.0};
    Parameter<float>  *x_center_drift, *y_center_drift, *crop_height_factor, *crop_width_factor;
    std::vector<size_t> x1_arr_val, y1_arr_val, croph_arr_val, cropw_arr_val;
    Parameter<float>* default_x_drift();
    Parameter<float>* default_y_drift();
    Parameter<float>* default_crop_height_factor();
    Parameter<float>* default_crop_width_factor();
};

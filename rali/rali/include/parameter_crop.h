/*
Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once
#include <VX/vx_types.h>
#include "parameter_factory.h"
enum class RaliCropType
{
    RALICROP = 0,
    RANDOMCROP,
};

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
    CropParam(unsigned int batch_size): batch_size(batch_size), _random(false)
    {
        x_drift_factor     = default_x_drift_factor();
        y_drift_factor     = default_y_drift_factor();
    }
    void set_image_dimensions( const std::vector<uint32_t>& in_width_, const std::vector<uint32_t>& in_height_)
    {
        if(in_width_.size() != in_width.size() || in_height.size() != in_height_.size())
            THROW("wrong input width = "+ TOSTR(in_width.size())+" or height size = "+TOSTR(in_height_.size()))
        in_width  = in_width_;
        in_height = in_height_;
    }
    void set_random() {_random = true;}
    void set_x_drift_factor(Parameter<float>* x_drift);
    void set_y_drift_factor(Parameter<float>* y_drift);
    std::vector<uint32_t> in_width, in_height;
    unsigned int  x1, y1, x2, y2;
    const unsigned int batch_size;
    void set_batch_size(unsigned int batch_size);
    vx_array x1_arr, y1_arr, croph_arr, cropw_arr, x2_arr, y2_arr;
    void array_init();
    void create_array(std::shared_ptr<Graph> graph);
    virtual void update_array() {};
    Parameter<float> * get_x_drift_factor() {return x_drift_factor;}
    Parameter<float> * get_y_drift_factor() {return y_drift_factor;}
    std::vector<uint32_t> get_x1_arr_val() {return x1_arr_val;}
    std::vector<uint32_t> get_y1_arr_val() {return y1_arr_val;}
    std::vector<uint32_t> get_croph_arr_val() {return croph_arr_val;}
    std::vector<uint32_t> get_cropw_arr_val() {return cropw_arr_val;}
    void get_crop_dimensions(std::vector<uint32_t> &crop_w_dim, std::vector<uint32_t> &crop_h_dim);
protected:
    constexpr static float CROP_X_DRIFT_RANGE [2]  = {0.01, 0.99};
    constexpr static float CROP_Y_DRIFT_RANGE [2]  = {0.01, 0.99};
    Parameter<float>  *x_drift_factor, *y_drift_factor;
    Parameter<float>* default_x_drift_factor();
    Parameter<float>* default_y_drift_factor();
    std::vector<uint32_t> x1_arr_val, y1_arr_val, croph_arr_val, cropw_arr_val, x2_arr_val, y2_arr_val;
    bool _random;
    virtual void fill_crop_dims(){};
    void update_crop_array();
};

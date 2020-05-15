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
    void set_image_dimensions(const std::vector<uint32_t>& in_width_, const std::vector<uint32_t>& in_height_)
    {
        if(in_width_.size() != in_width.size() || in_height.size() != in_height_.size())
            THROW("wrong input width = "+ TOSTR(in_width.size())+" or height size = "+TOSTR(in_height_.size()))
            in_width = in_width_;
        in_height =  in_height_;
        for(size_t image_idx = 0; image_idx < in_width_.size(); image_idx++)
        {
            x1[image_idx] = 0;
            x2[image_idx] = in_width_[image_idx] - 1;
            y1[image_idx] = 0;
            y2[image_idx] = in_height_[image_idx] - 1;
        }


    }
    void set_area_coeff(Parameter<float>* area);
    void set_x_drift(Parameter<float>* x_drift);
    void set_y_drift(Parameter<float>* y_drift);
    void set_aspect_ratio_coeff(Parameter<float>* aspect_ratio);
    std::vector<uint32_t> in_width, in_height;
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
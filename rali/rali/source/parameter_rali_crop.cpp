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

#include <cmath>
#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include <graph.h>
#include "parameter_rali_crop.h"
#include "commons.h"

void RaliCropParam::set_crop_height_factor(Parameter<float>* crop_h_factor)
{
    if(!crop_h_factor)
        return ;
    ParameterFactory::instance()->destroy_param(crop_height_factor);
    crop_height_factor = crop_h_factor;
}

void RaliCropParam::set_crop_width_factor(Parameter<float>* crop_w_factor)
{
    if(!crop_w_factor)
        return ;
    ParameterFactory::instance()->destroy_param(crop_width_factor);
    crop_width_factor = crop_w_factor;
}

void RaliCropParam::update_array()
{
    fill_crop_dims();
    update_crop_array();
}

void RaliCropParam::fill_crop_dims()
{
    int cropw_temp = 0, croph_temp = 0;
    float crop_h_factor_, crop_w_factor_, x_drift, y_drift;
    for(uint img_idx =0; img_idx < batch_size; img_idx++)
    {
        if(!(_random))
        {
            if(crop_w > in_width[img_idx])   { cropw_temp = in_width[img_idx]; }
		    else { cropw_temp = crop_w; }
            if(crop_h >= in_height[img_idx]) { croph_temp = in_height[img_idx]; }
		    else { croph_temp = crop_h; }
            cropw_arr_val[img_idx] = cropw_temp;
            croph_arr_val[img_idx] = croph_temp;
        }
        else
        {
            crop_height_factor->renew();
            crop_h_factor_ = crop_height_factor->get();
            crop_width_factor->renew();
            crop_w_factor_ = crop_width_factor->get();   
            cropw_arr_val[img_idx] = static_cast<size_t> (crop_w_factor_ * in_width[img_idx]);
            croph_arr_val[img_idx] = static_cast<size_t> (crop_h_factor_ * in_height[img_idx]);
        }
        x_drift_factor->renew();
        y_drift_factor->renew();
        y_drift_factor->renew();
        x_drift = x_drift_factor->get();
        y_drift = y_drift_factor->get();
        x1_arr_val[img_idx] = static_cast<size_t>(x_drift * (in_width[img_idx]  - cropw_arr_val[img_idx]));
        y1_arr_val[img_idx] = static_cast<size_t>(y_drift * (in_height[img_idx] - croph_arr_val[img_idx]));
        x2_arr_val[img_idx] = x1_arr_val[img_idx] + cropw_arr_val[img_idx];
        y2_arr_val[img_idx] = y1_arr_val[img_idx] + croph_arr_val[img_idx];
    } 
}

Parameter<float> *RaliCropParam::default_crop_height_factor()
{
    return ParameterFactory::instance()->create_uniform_float_rand_param(CROP_HEIGHT_FACTOR_RANGE[0],
                                                                         CROP_HEIGHT_FACTOR_RANGE[1])->core;
}

Parameter<float> *RaliCropParam::default_crop_width_factor()
{
    return ParameterFactory::instance()->create_uniform_float_rand_param(CROP_WIDTH_FACTOR_RANGE[0],
                                                                         CROP_WIDTH_FACTOR_RANGE[1])->core;
}

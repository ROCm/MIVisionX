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
#include "parameter_crop_resize.h"
#include "commons.h"


void RandomCropResizeParam::set_area_coeff(Parameter<float>* area)
{
    if(!area)
    { return; }

    ParameterFactory::instance()->destroy_param(area_coeff);
    area_coeff = area;
}

void RandomCropResizeParam::set_aspect_ratio_coeff(Parameter<float>* aspect_ratio)
{
    if(!aspect_ratio)
    { return; }

    ParameterFactory::instance()->destroy_param(aspect_ratio_coeff);
    aspect_ratio_coeff = aspect_ratio;
}
void RandomCropResizeParam::set_x_drift(Parameter<float>* x_drift)
{
    if(!x_drift)
    { return; }

    ParameterFactory::instance()->destroy_param(x_center_drift);
    x_center_drift = x_drift;
}
void RandomCropResizeParam::set_y_drift(Parameter<float>* y_drift)
{
    if(!y_drift)
    { return; }

    ParameterFactory::instance()->destroy_param(y_center_drift);
    y_center_drift = y_drift;
}

void RandomCropResizeParam::create_array(std::shared_ptr<Graph> graph)
{
    x1_arr_val.resize(batch_size);
    y1_arr_val.resize(batch_size);
    x2_arr_val.resize(batch_size);
    y2_arr_val.resize(batch_size);
    in_height.resize(batch_size);
    in_width.resize(batch_size);
    x1.resize(batch_size);
    x2.resize(batch_size);
    y1.resize(batch_size);
    y2.resize(batch_size);
    x1_arr = vxCreateArray(vxGetContext((vx_reference)graph->get()), VX_TYPE_UINT64,batch_size);
    y1_arr = vxCreateArray(vxGetContext((vx_reference)graph->get()), VX_TYPE_UINT64,batch_size);
    x2_arr = vxCreateArray(vxGetContext((vx_reference)graph->get()), VX_TYPE_UINT64,batch_size);
    y2_arr = vxCreateArray(vxGetContext((vx_reference)graph->get()), VX_TYPE_UINT64,batch_size);
    vxAddArrayItems(x1_arr,batch_size, x1_arr_val.data(), sizeof(size_t));
    vxAddArrayItems(y1_arr,batch_size, y1_arr_val.data(), sizeof(size_t));
    vxAddArrayItems(x2_arr,batch_size, x2_arr_val.data(), sizeof(size_t));
    vxAddArrayItems(y2_arr,batch_size, y2_arr_val.data(), sizeof(size_t));
    update_array();
}

void RandomCropResizeParam::update_array()
{
    vx_status status = VX_SUCCESS;
    for (uint img_idx = 0; img_idx < batch_size; img_idx++)
    {
        area_coeff->renew();
        float area = area_coeff->get();
        aspect_ratio_coeff->renew();
        float aspect_ratio_temp = aspect_ratio_coeff->get();
        x_center_drift->renew();
        float x_center = x_center_drift->get();
        y_center_drift->renew();
        y_center_drift->renew();
        float y_center = y_center_drift->get();
        // std::cerr<<"\n area :: "<<area<<"\t ::"<<aspect_ratio_temp<<"\t ::"<<x_center<<"\t ::"<<y_center<<"\n";
        calculate_area_cmn(img_idx,
                           area,
                           x_center,
                           y_center, aspect_ratio_temp);
        x1_arr_val[img_idx] = x1[img_idx];
        y1_arr_val[img_idx] = y1[img_idx];
        x2_arr_val[img_idx] = x2[img_idx];
        y2_arr_val[img_idx] = y2[img_idx];
        // std::cerr<<"\n x1 ::"<<x1<<"\t ::"<<x2<<"\t ::"<<y1<<"\t ::"<<y2;
    }
    status = vxCopyArrayRange((vx_array)x1_arr, 0, batch_size, sizeof(size_t), x1_arr_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != VX_SUCCESS)
    { WRN("ERROR: vxCopyArrayRange x1_arr failed " +TOSTR(status)); }
    status = vxCopyArrayRange((vx_array)y1_arr, 0, batch_size, sizeof(size_t), y1_arr_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != VX_SUCCESS)
    { WRN("ERROR: vxCopyArrayRange x1_arr failed " +TOSTR(status)); }
    status = vxCopyArrayRange((vx_array)x2_arr, 0, batch_size, sizeof(size_t), x2_arr_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != VX_SUCCESS)
    { WRN("ERROR: vxCopyArrayRange x1_arr failed " +TOSTR(status)); }
    status = vxCopyArrayRange((vx_array)y2_arr, 0, batch_size, sizeof(size_t), y2_arr_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != VX_SUCCESS)
    { WRN("ERROR: vxCopyArrayRange x1_arr failed " +TOSTR(status)); }
}

void RandomCropResizeParam::update_array_for_cmn() // For crop mirro normalize
{
    vx_status status = VX_SUCCESS;
    for (uint img_idx = 0; img_idx < batch_size; img_idx++)
    {
        area_coeff->renew();
        aspect_ratio_coeff->renew();
        float area = area_coeff->get();
        float aspect_ratio_temp = aspect_ratio_coeff->get();
        x_center_drift->renew();
        float x_center = x_center_drift->get();
        y_center_drift->renew();
        y_center_drift->renew();
        float y_center = y_center_drift->get();
        calculate_area_cmn(img_idx,
                           area,
                           x_center,
                           y_center, aspect_ratio_temp);
        x1_arr_val[img_idx] = x1[img_idx];
        y1_arr_val[img_idx] = y1[img_idx];
        x2_arr_val[img_idx] = x2[img_idx] - x1[img_idx] ;
        y2_arr_val[img_idx] = y2[img_idx] - y1[img_idx] ;

    }
    status = vxCopyArrayRange((vx_array)x1_arr, 0, batch_size, sizeof(size_t), x1_arr_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != VX_SUCCESS)
    { WRN("ERROR: vxCopyArrayRange x1_arr failed " +TOSTR(status)); }
    status = vxCopyArrayRange((vx_array)y1_arr, 0, batch_size, sizeof(size_t), y1_arr_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != VX_SUCCESS)
    { WRN("ERROR: vxCopyArrayRange x1_arr failed " +TOSTR(status)); }
    status = vxCopyArrayRange((vx_array)x2_arr, 0, batch_size, sizeof(size_t), x2_arr_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != VX_SUCCESS)
    { WRN("ERROR: vxCopyArrayRange x1_arr failed " +TOSTR(status)); }
    status = vxCopyArrayRange((vx_array)y2_arr, 0, batch_size, sizeof(size_t), y2_arr_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != VX_SUCCESS)
    { WRN("ERROR: vxCopyArrayRange x1_arr failed " +TOSTR(status)); }
}

void RandomCropResizeParam::calculate_area_cmn(unsigned image_idx, float area_coeff_, float x_center_drift_, float y_center_drift_, float aspect_ratio_)
{

// +-----------------------------------------> X direction
// |  ___________________________________
// |  |   (x1,y1)      |                |
// |  |    +-----------|-----------+    |
// |  |    |           |           |    |
// |  -----------------o-----------------
// |  |    |           |           |    |
// |  |    +-----------|-----------+    |
// |  |                |        (x2,y2) |
// |  +++++++++++++++++++++++++++++++++++
// |
// V Y directoin

    auto bound = [](float arg, float min, float max)
    {
        if( arg < min)
        { return min; }
        if( arg > max)
        { return max; }
        return arg;
    };

    auto y_center = in_height[image_idx] / 2;
    auto x_center = in_width[image_idx] / 2;


    auto temp_aspect_ratio = aspect_ratio_;

    float length_coeff = std::sqrt(bound(area_coeff_, RandomCropResizeParam::MIN_RANDOM_AREA_COEFF, 1.0));

    auto cropped_width = (size_t)(length_coeff  * (float)in_width[image_idx] );
    auto cropped_height = (size_t)(cropped_width / temp_aspect_ratio) ;

    // This will adjust to the input aspect ratio if crops are going out of bound - aspect ration will be relaxed
    if (cropped_width > in_width[image_idx] || cropped_height > in_height[image_idx])
    {
        temp_aspect_ratio = ((float)in_width[image_idx]  / in_height[image_idx] );
        cropped_width = (size_t)(length_coeff  * (float)in_width[image_idx] );
        cropped_height = (size_t)( (float)(cropped_width) / temp_aspect_ratio) ;
    }

    size_t y_max_drift = (in_height[image_idx] - cropped_height) / 2;
    size_t x_max_drift = (in_width[image_idx] - cropped_width ) / 2;


    size_t no_drift_y1 = y_center - cropped_height/2;
    size_t no_drift_x1 = x_center - cropped_width/2;


    float x_drift_coeff = bound(x_center_drift_, -1.0, 1.0);// in [-1 1] range
    float y_drift_coeff = bound(y_center_drift_, -1.0, 1.0);// in [-1 1] range


    x1[image_idx] = (size_t)((float)no_drift_x1 + x_drift_coeff * (float)x_max_drift);
    y1[image_idx] = (size_t)((float)no_drift_y1 + y_drift_coeff * (float)y_max_drift);

    x1[image_idx] = x_center - cropped_width/2; // ROI centric
    y1[image_idx] = y_center - cropped_height/2; // ROI centric


    x2[image_idx] = x1[image_idx] + cropped_width ;
    y2[image_idx] = y1[image_idx] + cropped_height ;

    auto check_bound = [](int arg, int min, int max)
    {
        return arg < min || arg > max;
    };

    if(check_bound(x1[image_idx], 0, in_width[image_idx]) || check_bound(x2[image_idx], 0, in_width[image_idx]) || check_bound(y1[image_idx], 0, in_height[image_idx]) || check_bound(y2[image_idx], 0, in_height[image_idx]))
        // TODO: proper action required here
        WRN("Wrong crop area calculation")
    }
Parameter<float> *RandomCropResizeParam::default_area()
{
    return ParameterFactory::instance()->create_uniform_float_rand_param(CROP_AREA_RANGE[0],
            CROP_AREA_RANGE[1])->core;
}

Parameter<float> *RandomCropResizeParam::default_aspect_ratio()
{
    return ParameterFactory::instance()->create_uniform_float_rand_param(CROP_ASPECT_RATIO[0],
            CROP_ASPECT_RATIO[1])->core;
}

Parameter<float> *RandomCropResizeParam::default_x_drift()
{
    return ParameterFactory::instance()->create_uniform_float_rand_param(CROP_AREA_X_DRIFT_RANGE[0],
            CROP_AREA_X_DRIFT_RANGE[1])->core;
}

Parameter<float> *RandomCropResizeParam::default_y_drift()
{
    return ParameterFactory::instance()->create_uniform_float_rand_param(CROP_AREA_Y_DRIFT_RANGE[0],
            CROP_AREA_Y_DRIFT_RANGE[1])->core;
}
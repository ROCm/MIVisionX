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


#include <node_warp_affine.h>
#include "node_exposure.h"
#include "node_vignette.h"
#include "node_jitter.h"
#include "node_snp_noise.h"
#include "node_snow.h"
#include "node_rain.h"
#include "node_color_temperature.h"
#include "node_fog.h"
#include "node_pixelate.h"
#include "node_lens_correction.h"
#include "node_gamma.h"
#include "node_flip.h"
#include "node_crop_resize.h"
#include "node_brightness.h"
#include "node_contrast.h"
#include "node_blur.h"
#include "node_fisheye.h"
#include "node_blend.h"
#include "node_resize.h"
#include "node_rotate.h"
#include "node_color_twist.h"
#include "node_hue.h"
#include "node_saturation.h"
#include "node_crop_mirror_normalize.h"
#include "node_resize_crop_mirror.h"
#include "node_ssd_random_crop.h"
#include "node_crop.h"
#include "node_random_crop.h"
#include "node_copy.h"
#include "node_nop.h"
#include "meta_node_crop_mirror_normalize.h"
#include "meta_node_resize.h"
#include "meta_node_crop_resize.h"
#include "meta_node_crop.h"
#include "meta_node_resize_crop_mirror.h"
#include "meta_node_rotate.h"
#include "meta_node_ssd_random_crop.h"
#include "meta_node_flip.h"

#include "commons.h"
#include "context.h"
#include "rali_api.h"



RaliImage  RALI_API_CALL
raliRotate(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output,
        RaliFloatParam p_angle,
        unsigned dest_width,
        unsigned dest_height)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto angle = static_cast<FloatParam*>(p_angle);
    try
    {
        if(dest_width == 0 || dest_height == 0)
        {
            dest_width = input->info().width();
            dest_height = input->info().height_single();
        }
        // For the rotate node, user can create an image with a different width and height
        ImageInfo output_info = input->info();
        output_info.width(dest_width);
        output_info.height(dest_height);

        output = context->master_graph->create_image(output_info, is_output);

        // If the user has provided the output size the dimension of all the images after this node will be fixed and equal to that size
        if(dest_width != 0 && dest_height != 0)
            output->reset_image_roi();
        std::shared_ptr<RotateNode> rotate_node =  context->master_graph->add_node<RotateNode>({input}, {output});
        rotate_node->init(angle);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<RotateMetaNode,RotateNode>(rotate_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage  RALI_API_CALL
raliRotateFixed(
        RaliContext p_context,
        RaliImage p_input,
        float angle,
        bool is_output,
        unsigned dest_width,
        unsigned dest_height)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {
        if(dest_width == 0 || dest_height == 0)
        {
            dest_width = input->info().width();
            dest_height = input->info().height_single();
        }
        // For the rotate node, user can create an image with a different width and height
        ImageInfo output_info = input->info();
        output_info.width(dest_width);
        output_info.height(dest_height);

        output = context->master_graph->create_image(output_info, is_output);

        // If the user has provided the output size the dimension of all the images after this node will be fixed and equal to that size
        if(dest_width != 0 && dest_height != 0)
            output->reset_image_roi();

        std::shared_ptr<RotateNode> rotate_node =  context->master_graph->add_node<RotateNode>({input}, {output});
        rotate_node->init(angle);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<RotateMetaNode,RotateNode>(rotate_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliFlip(
        RaliContext p_context,
        RaliImage p_input,
        RaliFlipAxis axis,
        bool is_output)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {
        output = context->master_graph->create_image(input->info(), is_output);
        std::shared_ptr<FlipNode> flip_node =  context->master_graph->add_node<FlipNode>({input}, {output});
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<FlipMetaNode,FlipNode>(flip_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


RaliImage RALI_API_CALL
raliGamma(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output,
        RaliFloatParam p_alpha)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    try
    {
        output = context->master_graph->create_image(input->info(), is_output);
        context->master_graph->add_node<GammaNode>({input}, {output})->init(alpha);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliGammaFixed(
        RaliContext p_context,
        RaliImage p_input,
        float alpha,
        bool is_output)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")

    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {
        output = context->master_graph->create_image(input->info(), is_output);
        context->master_graph->add_node<GammaNode>({input}, {output})->init(alpha);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliHue(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output,
        RaliFloatParam p_hue)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")

    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto hue = static_cast<FloatParam*>(p_hue);
    try
    {
        output = context->master_graph->create_image(input->info(), is_output);
        context->master_graph->add_node<HueNode>({input}, {output})->init(hue);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliHueFixed(
        RaliContext p_context,
        RaliImage p_input,
        float hue,
        bool is_output)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {
        output = context->master_graph->create_image(input->info(), is_output);
        context->master_graph->add_node<HueNode>({input}, {output})->init(hue);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliSaturation(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output,
        RaliFloatParam p_sat)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto sat = static_cast<FloatParam*>(p_sat);
    try
    {
        output = context->master_graph->create_image(input->info(), is_output);
        context->master_graph->add_node<SatNode>({input}, {output})->init(sat);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliSaturationFixed(
        RaliContext p_context,
        RaliImage p_input,
        float sat,
        bool is_output)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {
        output = context->master_graph->create_image(input->info(), is_output);

        context->master_graph->add_node<SatNode>({input}, {output})->init(sat);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage  RALI_API_CALL
raliCropResize(
        RaliContext p_context,
        RaliImage p_input,
        unsigned dest_width, unsigned dest_height,
        bool is_output,
        RaliFloatParam p_area,
        RaliFloatParam p_aspect_ratio,
        RaliFloatParam p_x_center_drift,
        RaliFloatParam p_y_center_drift)
{
    Image* output = nullptr;
    if(!p_input || !p_context )
        THROW("Null values passed as input")
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto area = static_cast<FloatParam*>(p_area);
    auto aspect_ratio = static_cast<FloatParam*>(p_aspect_ratio);
    auto x_center_drift = static_cast<FloatParam*>(p_x_center_drift);
    auto y_center_drift = static_cast<FloatParam*>(p_y_center_drift);
    try
    {
        if(dest_width == 0 || dest_height == 0)
            THROW("CropResize node needs tp receive non-zero destination dimensions")
        // For the crop resize node, user can create an image with a different width and height
        ImageInfo output_info = input->info();

        output_info.width(dest_width);
        output_info.height(dest_height);
        output = context->master_graph->create_image(output_info, is_output);

        // For the nodes that user provides the output size the dimension of all the images after this node will be fixed and equal to that size
        output->reset_image_roi();

        std::shared_ptr<CropResizeNode> crop_resize_node =  context->master_graph->add_node<CropResizeNode>({input}, {output});
        crop_resize_node->init(area, aspect_ratio, x_center_drift, y_center_drift);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropResizeMetaNode,CropResizeNode>(crop_resize_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


RaliImage  RALI_API_CALL
raliCropResizeFixed(
        RaliContext p_context,
        RaliImage p_input,
        unsigned dest_width, unsigned dest_height,
        bool is_output,
        float area,
        float aspect_ratio,
        float x_center_drift,
        float y_center_drift)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {
        if(dest_width == 0 || dest_height == 0)
            THROW("CropResize node needs tp receive non-zero destination dimensions")
        // For the crop resize node, user can create an image with a different width and height
        ImageInfo output_info = input->info();

        output_info.width(dest_width);
        output_info.height(dest_height);
        output = context->master_graph->create_image(output_info, is_output);

        // user provides the output size and the dimension of all the images after this node will be fixed and equal to that size
        output->reset_image_roi();

        std::shared_ptr<CropResizeNode> crop_resize_node =  context->master_graph->add_node<CropResizeNode>({input}, {output});
        crop_resize_node->init(area, aspect_ratio, x_center_drift, y_center_drift);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropResizeMetaNode,CropResizeNode>(crop_resize_node);

    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage  RALI_API_CALL
raliResize(
        RaliContext p_context,
        RaliImage p_input,
        unsigned dest_width,
        unsigned dest_height,
        bool is_output)
{
    Image* output = nullptr;
    if(!p_input || !p_context || dest_width == 0 || dest_height == 0)
        THROW("Null values passed as input")
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {
        // For the resize node, user can create an image with a different width and height
        ImageInfo output_info = input->info();
        output_info.width(dest_width);
        output_info.height(dest_height);

        output = context->master_graph->create_image(output_info, is_output);

        // For the nodes that user provides the output size the dimension of all the images after this node will be fixed and equal to that size
        output->reset_image_roi();

        std::shared_ptr<ResizeNode> resize_node =  context->master_graph->add_node<ResizeNode>({input}, {output});
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<ResizeMetaNode,ResizeNode>(resize_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliBrightness(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output,
        RaliFloatParam p_alpha,
        RaliIntParam p_beta)
{
    Image* output = nullptr;
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    auto beta = static_cast<IntParam*>(p_beta);
    try
    {

        output = context->master_graph->create_image(input->info(), is_output);

        context->master_graph->add_node<BrightnessNode>({input}, {output})->init(alpha, beta);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliBrightnessFixed(
        RaliContext p_context,
        RaliImage p_input,
        float alpha,
        int beta,
        bool is_output)
{
    Image* output = nullptr;
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {
        output = context->master_graph->create_image(input->info(), is_output);
        context->master_graph->add_node<BrightnessNode>({input}, {output})->init(alpha, beta);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliBlur(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output,
        RaliIntParam p_sdev)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto sdev = static_cast<IntParam*>(p_sdev);
    try
    {
        output = context->master_graph->create_image(input->info(), is_output);
        context->master_graph->add_node<BlurNode>({input}, {output})->init(sdev);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliBlurFixed(
        RaliContext p_context,
        RaliImage p_input,
        int sdev,
        bool is_output)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {
        output = context->master_graph->create_image(input->info(), is_output);
        context->master_graph->add_node<BlurNode>({input}, {output})->init(sdev);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage  RALI_API_CALL
raliBlend(
        RaliContext p_context,
        RaliImage p_input1,
        RaliImage p_input2,
        bool is_output,
        RaliFloatParam p_ratio)
{
    Image* output = nullptr;
    if(!p_input1 || !p_input2 || !p_context)
        THROW("Null values passed as input")
    auto context = static_cast<Context*>(p_context);
    auto input1 = static_cast<Image*>(p_input1);
    auto input2 = static_cast<Image*>(p_input2);
    auto ratio = static_cast<FloatParam*>(p_ratio);
    try
    {
        if(!(input1->info() == input2->info()))
            THROW("Input images to the blend operation must have the same info")
        output = context->master_graph->create_image(input1->info(), is_output);
        context->master_graph->add_node<BlendNode>({input1, input2}, {output})->init(ratio);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage  RALI_API_CALL
raliBlendFixed(
        RaliContext p_context,
        RaliImage p_input1,
        RaliImage p_input2,
        float ratio,
        bool is_output)
{
    Image* output = nullptr;
    if(!p_input1 || !p_input2 || !p_context)
        THROW("Null values passed as input")
    auto context = static_cast<Context*>(p_context);
    auto input1 = static_cast<Image*>(p_input1);
    auto input2 = static_cast<Image*>(p_input2);
    try
    {

        if(!(input1->info() == input2->info()))
            THROW("Input images to the blend operation must have the same info")

        output = context->master_graph->create_image(input1->info(), is_output);

        context->master_graph->add_node<BlendNode>({input1, input2}, {output})->init(ratio);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


RaliImage  RALI_API_CALL
raliWarpAffine(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output,
        unsigned dest_height, unsigned dest_width,
        RaliFloatParam p_x0, RaliFloatParam p_x1,
        RaliFloatParam p_y0, RaliFloatParam p_y1,
        RaliFloatParam p_o0, RaliFloatParam p_o1)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto x0 = static_cast<FloatParam*>(p_x0);
    auto x1 = static_cast<FloatParam*>(p_x1);
    auto y0 = static_cast<FloatParam*>(p_y0);
    auto y1 = static_cast<FloatParam*>(p_y1);
    auto o0 = static_cast<FloatParam*>(p_o0);
    auto o1 = static_cast<FloatParam*>(p_o1);
    try
    {
        if(dest_width == 0 || dest_height == 0)
        {
            dest_width = input->info().width();
            dest_height = input->info().height_single();
        }
        // For the warp affine node, user can create an image with a different width and height
        ImageInfo output_info = input->info();
        output_info.width(dest_width);
        output_info.height(dest_height);

        output = context->master_graph->create_image(output_info, is_output);

        // If the user has provided the output size the dimension of all the images after this node will be fixed and equal to that size
        if(dest_width != 0 && dest_height != 0)
            output->reset_image_roi();

        context->master_graph->add_node<WarpAffineNode>({input}, {output})->init(x0, x1, y0, y1, o0, o1);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage  RALI_API_CALL
raliWarpAffineFixed(
        RaliContext p_context,
        RaliImage p_input,
        float x0, float x1,
        float y0, float y1,
        float o0, float o1,
        bool is_output,
        unsigned int dest_height,
        unsigned int dest_width)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {

        if(dest_width == 0 || dest_height == 0)
        {
            dest_width = input->info().width();
            dest_height = input->info().height_single();
        }
        // For the warp affine node, user can create an image with a different width and height
        ImageInfo output_info = input->info();
        output_info.width(dest_width);
        output_info.height(dest_height);

        output = context->master_graph->create_image(input->info(), is_output);

        // If the user has provided the output size the dimension of all the images after this node will be fixed and equal to that size
        if(dest_width != 0 && dest_height != 0)
            output->reset_image_roi();

        context->master_graph->add_node<WarpAffineNode>({input}, {output})->init(x0, x1, y0, y1, o0, o1);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage  RALI_API_CALL
raliFishEye(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {

        output = context->master_graph->create_image(input->info(), is_output);

        context->master_graph->add_node<FisheyeNode>({input}, {output});
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliVignette(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output,
        RaliFloatParam p_sdev)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto sdev = static_cast<FloatParam*>(p_sdev);
    try
    {

        output = context->master_graph->create_image(input->info(), is_output);

        context->master_graph->add_node<VignetteNode>({input}, {output})->init(sdev);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliVignetteFixed(
        RaliContext p_context,
        RaliImage p_input,
        float sdev,
        bool is_output)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {
        output = context->master_graph->create_image(input->info(), is_output);

        context->master_graph->add_node<VignetteNode>({input}, {output})->init(sdev);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliJitter(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output,
        RaliIntParam p_kernel_size)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto kernel_size = static_cast<IntParam*>(p_kernel_size);
    try
    {
        if(!input || !context)
            THROW("Null values passed as input")

        output = context->master_graph->create_image(input->info(), is_output);

        context->master_graph->add_node<JitterNode>({input}, {output})->init(kernel_size);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliJitterFixed(
        RaliContext p_context,
        RaliImage p_input,
        int kernel_size,
        bool is_output)
{

    Image* output = nullptr;
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {

        output = context->master_graph->create_image(input->info(), is_output);

        context->master_graph->add_node<JitterNode>({input}, {output})->init(kernel_size);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


RaliImage RALI_API_CALL
raliSnPNoise(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output,
        RaliFloatParam p_sdev)
{
    Image* output = nullptr;
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto sdev = static_cast<FloatParam*>(p_sdev);
    try
    {

        output = context->master_graph->create_image(input->info(), is_output);

        context->master_graph->add_node<SnPNoiseNode>({input}, {output})->init(sdev);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliSnPNoiseFixed(
        RaliContext p_context,
        RaliImage p_input,
        float sdev,
        bool is_output)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {

        output = context->master_graph->create_image(input->info(), is_output);

        context->master_graph->add_node<SnPNoiseNode>({input}, {output})->init(sdev);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliFlip(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output,
        RaliIntParam p_flip_axis)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto flip_axis = static_cast<IntParam*>(p_flip_axis);
    try
    {

        output = context->master_graph->create_image(input->info(), is_output);

        context->master_graph->add_node<FlipNode>({input}, {output})->init(flip_axis);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliFlipFixed(
        RaliContext p_context,
        RaliImage p_input,
        int flip_axis,
        bool is_output)
{
    Image* output = nullptr;
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {

        output = context->master_graph->create_image(input->info(), is_output);

        context->master_graph->add_node<FlipNode>({input}, {output})->init(flip_axis);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


RaliImage RALI_API_CALL
raliContrast(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output,
        RaliIntParam p_min,
        RaliIntParam p_max)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto min = static_cast<IntParam*>(p_min);
    auto max = static_cast<IntParam*>(p_max);
    try
    {

        output = context->master_graph->create_image(input->info(), is_output);

        context->master_graph->add_node<RaliContrastNode>({input}, {output})->init(min, max);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliContrastFixed(
        RaliContext p_context,
        RaliImage p_input,
        unsigned min,
        unsigned max,
        bool is_output)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {

        output = context->master_graph->create_image(input->info(), is_output);

        context->master_graph->add_node<RaliContrastNode>({input}, {output})->init(min, max);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliSnow(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output,
        RaliFloatParam p_shift)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto shift = static_cast<FloatParam*>(p_shift);
    try
    {

        output = context->master_graph->create_image(input->info(), is_output);

        context->master_graph->add_node<SnowNode>({input}, {output})->init(shift);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliSnowFixed(
        RaliContext p_context,
        RaliImage p_input,
        float shift,
        bool is_output)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {

        output = context->master_graph->create_image(input->info(), is_output);

        context->master_graph->add_node<SnowNode>({input}, {output})->init(shift);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliRain(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output,
        RaliFloatParam p_rain_value,
        RaliIntParam p_rain_width,
        RaliIntParam p_rain_height,
        RaliFloatParam p_rain_transparency)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto rain_width = static_cast<IntParam*>(p_rain_width);
    auto rain_height = static_cast<IntParam*>(p_rain_height);
    auto rain_transparency = static_cast<FloatParam*>(p_rain_transparency);
    auto rain_value = static_cast<FloatParam*>(p_rain_value);
    try
    {

        output = context->master_graph->create_image(input->info(), is_output);

        context->master_graph->add_node<RainNode>({input}, {output})->init(rain_value, rain_width, rain_height, rain_transparency);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliRainFixed(
        RaliContext p_context,
        RaliImage p_input,
        float rain_value,
        int rain_width,
        int rain_height,
        float rain_transparency,
        bool is_output)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {
        output = context->master_graph->create_image(input->info(), is_output);
        context->master_graph->add_node<RainNode>({input}, {output})->init(rain_value, rain_width, rain_height, rain_transparency);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliColorTemp(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output,
        RaliIntParam p_adj_value_param)
{
    if(!p_input || !p_context)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto adj_value_param = static_cast<IntParam*>(p_adj_value_param);
    try
    {
        output = context->master_graph->create_image(input->info(), is_output);
        context->master_graph->add_node<ColorTemperatureNode>({input}, {output})->init(adj_value_param);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliColorTempFixed(
        RaliContext p_context,
        RaliImage p_input,
        int adj_value_param,
        bool is_output)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {

        output = context->master_graph->create_image(input->info(), is_output);

        context->master_graph->add_node<ColorTemperatureNode>({input}, {output})->init(adj_value_param);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}


RaliImage RALI_API_CALL
raliFog(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output,
        RaliFloatParam p_fog_param)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto fog_param = static_cast<FloatParam*>(p_fog_param);
    try
    {
        output = context->master_graph->create_image(input->info(), is_output);

        context->master_graph->add_node<FogNode>({input}, {output})->init(fog_param);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliFogFixed(
        RaliContext p_context,
        RaliImage p_input,
        float fog_param,
        bool is_output)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {

        output = context->master_graph->create_image(input->info(), is_output);

        context->master_graph->add_node<FogNode>({input}, {output})->init(fog_param);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage  RALI_API_CALL
raliPixelate(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {

        output = context->master_graph->create_image(input->info(), is_output);

        context->master_graph->add_node<PixelateNode>({input}, {output});
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliLensCorrection(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output,
        RaliFloatParam p_strength,
        RaliFloatParam p_zoom)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto strength = static_cast<FloatParam*>(p_strength);
    auto zoom = static_cast<FloatParam*>(p_zoom);
    try
    {
        output = context->master_graph->create_image(input->info(), is_output);

        context->master_graph->add_node<LensCorrectionNode>({input}, {output})->init(strength, zoom);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliLensCorrectionFixed(
        RaliContext p_context,
        RaliImage p_input,
        float strength,
        float zoom,
        bool is_output)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {
        output = context->master_graph->create_image(input->info(), is_output);
        context->master_graph->add_node<LensCorrectionNode>({input}, {output})->init(strength, zoom);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliExposure(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output,
        RaliFloatParam p_shift)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto shift = static_cast<FloatParam*>(p_shift);
    try
    {
        output = context->master_graph->create_image(input->info(), is_output);
        context->master_graph->add_node<ExposureNode>({input}, {output})->init(shift);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliExposureFixed(
        RaliContext p_context,
        RaliImage p_input,
        float shift,
        bool is_output)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {

        output = context->master_graph->create_image(input->info(), is_output);
        context->master_graph->add_node<ExposureNode>({input}, {output})->init(shift);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliColorTwist(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output,
        RaliFloatParam p_alpha,
        RaliFloatParam p_beta,
        RaliFloatParam p_hue,
        RaliFloatParam p_sat)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto alpha = static_cast<FloatParam*>(p_alpha);
    auto beta = static_cast<FloatParam*>(p_beta);
    auto hue = static_cast<FloatParam*>(p_hue);
    auto sat = static_cast<FloatParam*>(p_sat);
    try
    {
        output = context->master_graph->create_image(input->info(), is_output);
        context->master_graph->add_node<ColorTwistBatchNode>({input}, {output})->init(alpha, beta, hue, sat);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage RALI_API_CALL
raliColorTwistFixed(
        RaliContext p_context,
        RaliImage p_input,
        float alpha,
        float beta,
        float hue,
        float sat,
        bool is_output)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {
        output = context->master_graph->create_image(input->info(), is_output);
        context->master_graph->add_node<ColorTwistBatchNode>({input}, {output})->init(alpha, beta, hue, sat);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage
RALI_API_CALL raliCropMirrorNormalize(RaliContext p_context, RaliImage p_input, unsigned crop_depth, unsigned crop_height,
                                    unsigned crop_width, float start_x, float start_y, float start_z, std::vector<float> &mean,
                                    std::vector<float> &std_dev, bool is_output, RaliIntParam p_mirror)
{
    if(!p_context || !p_input )
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto mirror = static_cast<IntParam *>(p_mirror);
    float mean_acutal = 0, std_actual = 0; // Mean of vectors
    for(unsigned i = 0; i < mean.size(); i++)
    {
        mean_acutal += mean[i];
        std_actual  += std_dev[i];
    }
    mean_acutal /= mean.size();
    std_actual /= std_dev.size();

   try
    {
        if( crop_width == 0 || crop_height == 0)
            THROW("Null values passed as input")

        // For the crop mirror normalize resize node, user can create an image with a different width and height
        ImageInfo output_info = input->info();
        output_info.width(crop_width);
        output_info.height(crop_height);
        output = context->master_graph->create_image(output_info, is_output);
        // For the nodes that user provides the output size the dimension of all the images after this node will be fixed and equal to that size
        output->reset_image_roi();

        std::shared_ptr<CropMirrorNormalizeNode> cmn_node =  context->master_graph->add_node<CropMirrorNormalizeNode>({input}, {output});
        cmn_node->init(crop_height, crop_width, start_x, start_y, 0, 1 , mirror );
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropMirrorNormalizeMetaNode,CropMirrorNormalizeNode>(cmn_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return input; // Changed to input----------------IMPORTANT
}


RaliImage RALI_API_CALL
raliCrop(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output,
        RaliFloatParam p_crop_width,
        RaliFloatParam p_crop_height,
        RaliFloatParam p_crop_depth,
        RaliFloatParam p_crop_pox_x,
        RaliFloatParam p_crop_pos_y,
        RaliFloatParam p_crop_pos_z)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto crop_h = static_cast<FloatParam*>(p_crop_height);
    auto crop_w = static_cast<FloatParam*>(p_crop_width);
    auto x_drift = static_cast<FloatParam*>(p_crop_pox_x);
    auto y_drift = static_cast<FloatParam*>(p_crop_pos_y);

    try
    {
        ImageInfo output_info = input->info();
        output_info.width(input->info().width());
        output_info.height(input->info().height_single());
        output = context->master_graph->create_image(output_info, is_output);
        output->reset_image_roi();
        std::shared_ptr<CropNode> crop_node =  context->master_graph->add_node<CropNode>({input}, {output});
        crop_node->init(crop_h, crop_w, x_drift, y_drift);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropMetaNode,CropNode>(crop_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage  RALI_API_CALL
raliCropFixed(
        RaliContext p_context,
        RaliImage p_input,
        unsigned crop_width,
        unsigned crop_height,
        unsigned crop_depth,
        bool is_output,
        float crop_pos_x,
        float crop_pos_y,
        float crop_pos_z)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {
        if(crop_width == 0 || crop_height == 0 || crop_depth == 0)
            THROW("Crop node needs tp receive non-zero destination dimensions")
        // For the crop node, user can create an image with a different width and height
        ImageInfo output_info = input->info();
        output_info.width(crop_width);
        output_info.height(crop_height);
        output = context->master_graph->create_image(input->info(), is_output);
        output->reset_image_roi();
        std::shared_ptr<CropNode> crop_node =  context->master_graph->add_node<CropNode>({input}, {output});
        crop_node->init(crop_height, crop_width, crop_pos_x, crop_pos_y);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropMetaNode,CropNode>(crop_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage  RALI_API_CALL
raliCropCenterFixed(
        RaliContext p_context,
        RaliImage p_input,
        unsigned crop_width,
        unsigned crop_height,
        unsigned crop_depth,
        bool is_output)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {
        if(crop_width == 0 || crop_height == 0 || crop_depth == 0)
            THROW("Crop node needs tp receive non-zero destination dimensions")
        // For the crop node, user can create an image with a different width and height
        ImageInfo output_info = input->info();
        output_info.width(crop_width);
        output_info.height(crop_height);
        output = context->master_graph->create_image(input->info(), is_output);
        output->reset_image_roi();
        std::shared_ptr<CropNode> crop_node =  context->master_graph->add_node<CropNode>({input}, {output});
        crop_node->init(crop_height, crop_width);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<CropMetaNode,CropNode>(crop_node);
    }

    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage  RALI_API_CALL
raliResizeCropMirrorFixed(
        RaliContext p_context,
        RaliImage p_input,
        unsigned dest_width,
        unsigned dest_height,
        bool is_output,
        unsigned crop_h,
        unsigned crop_w,
        RaliIntParam p_mirror
        )
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto mirror = static_cast<IntParam *>(p_mirror);
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    try
    {
        if(dest_width == 0 || dest_height == 0)
            THROW("Crop Mirror node needs tp receive non-zero destination dimensions")
        // For the crop node, user can create an image with a different width and height
        ImageInfo output_info = input->info();
        output_info.width(dest_width);
        output_info.height(dest_height);
        output = context->master_graph->create_image(output_info, is_output);
        output->reset_image_roi();
        std::shared_ptr<ResizeCropMirrorNode> rcm_node =  context->master_graph->add_node<ResizeCropMirrorNode>({input}, {output});
        rcm_node->init(crop_h, crop_w, mirror);
        if (context->master_graph->meta_data_graph())
        context->master_graph->meta_add_node<ResizeCropMirrorMetaNode,ResizeCropMirrorNode>(rcm_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

extern "C"  RaliImage  RALI_API_CALL raliResizeCropMirror( RaliContext p_context, RaliImage p_input,
                                                           unsigned dest_width, unsigned dest_height,
                                                            bool is_output, RaliFloatParam p_crop_height,
                                                            RaliFloatParam p_crop_width, RaliIntParam p_mirror 
                                                            )
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto crop_h = static_cast<FloatParam*>(p_crop_height);
    auto crop_w = static_cast<FloatParam*>(p_crop_width);
    auto mirror  = static_cast<IntParam*>(p_mirror);
    try
    {
        if(dest_width == 0 || dest_height == 0)
            THROW("Crop Mirror node needs tp receive non-zero destination dimensions")
        // For the crop node, user can create an image with a different width and height
        ImageInfo output_info = input->info();
        output_info.width(dest_width);
        output_info.height(dest_height);
        output = context->master_graph->create_image(output_info, is_output);
        output->reset_image_roi();
        std::shared_ptr<ResizeCropMirrorNode> rcm_node =  context->master_graph->add_node<ResizeCropMirrorNode>({input}, {output});
        rcm_node->init(crop_h, crop_w, mirror);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<ResizeCropMirrorMetaNode,ResizeCropMirrorNode>(rcm_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

extern "C" RaliImage RALI_API_CALL
raliRandomCrop(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output,
        RaliFloatParam p_crop_area_factor,
        RaliFloatParam p_crop_aspect_ratio,
        RaliFloatParam p_crop_pox_x,
        RaliFloatParam p_crop_pos_y,
        int num_of_attempts)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto crop_area_factor  = static_cast<FloatParam*>(p_crop_area_factor);
    auto crop_aspect_ratio = static_cast<FloatParam*>(p_crop_aspect_ratio);
    auto x_drift = static_cast<FloatParam*>(p_crop_pox_x);
    auto y_drift = static_cast<FloatParam*>(p_crop_pos_y);

    try
    {
        ImageInfo output_info = input->info();
        output_info.width(input->info().width());
        output_info.height(input->info().height_single());
        output = context->master_graph->create_image(output_info, is_output);
        output->reset_image_roi();
        std::shared_ptr<RandomCropNode> crop_node =  context->master_graph->add_node<RandomCropNode>({input}, {output});
        crop_node->init(crop_area_factor, crop_aspect_ratio, x_drift, y_drift, num_of_attempts);
        // if (context->master_graph->meta_data_graph())
        //     context->master_graph->meta_add_node<SSDRandomCropMetaNode,RandomCropNode>(crop_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

extern "C" RaliImage RALI_API_CALL
raliSSDRandomCrop(
        RaliContext p_context,
        RaliImage p_input,
        bool is_output,
        RaliFloatParam p_threshold,
        RaliFloatParam p_crop_area_factor,
        RaliFloatParam p_crop_aspect_ratio,
        RaliFloatParam p_crop_pox_x,
        RaliFloatParam p_crop_pos_y,
        int num_of_attempts)
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);
    auto input = static_cast<Image*>(p_input);
    auto crop_area_factor  = static_cast<FloatParam*>(p_crop_area_factor);
    auto crop_aspect_ratio = static_cast<FloatParam*>(p_crop_aspect_ratio);
    auto x_drift = static_cast<FloatParam*>(p_crop_pox_x);
    auto y_drift = static_cast<FloatParam*>(p_crop_pos_y);

    try
    {
        ImageInfo output_info = input->info();
        output_info.width(input->info().width());
        output_info.height(input->info().height_single());
        output = context->master_graph->create_image(output_info, is_output);
        output->reset_image_roi();
        std::shared_ptr<SSDRandomCropNode> crop_node =  context->master_graph->add_node<SSDRandomCropNode>({input}, {output});
        crop_node->init(crop_area_factor, crop_aspect_ratio, x_drift, y_drift, num_of_attempts);
        if (context->master_graph->meta_data_graph())
            context->master_graph->meta_add_node<SSDRandomCropMetaNode,SSDRandomCropNode>(crop_node);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
    }
    return output;
}

RaliImage  RALI_API_CALL	
raliCopy(	
        RaliContext p_context,	
        RaliImage p_input,	
        bool is_output)	
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);	
    auto input = static_cast<Image*>(p_input);	
    try	
    {	
        output = context->master_graph->create_image(input->info(), is_output);
        context->master_graph->add_node<CopyNode>({input}, {output});
    }	
    catch(const std::exception& e)	
    {	
        context->capture_error(e.what());	
        ERR(e.what())	
    }	
    return output;	
}	

RaliImage  RALI_API_CALL	
raliNop(	
        RaliContext p_context,	
        RaliImage p_input,	
        bool is_output)	
{
    if(!p_context || !p_input)
        THROW("Null values passed as input")
    Image* output = nullptr;
    auto context = static_cast<Context*>(p_context);	
    auto input = static_cast<Image*>(p_input);	
    try	
    {	
        output = context->master_graph->create_image(input->info(), is_output);
        context->master_graph->add_node<NopNode>({input}, {output});
    }	
    catch(const std::exception& e)	
    {	
        context->capture_error(e.what());	
        ERR(e.what())	
    }	
    return output;	
}

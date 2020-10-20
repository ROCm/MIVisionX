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

#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_random_crop.h"
#include "exception.h"

RandomCropNode::RandomCropNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _dest_width(_outputs[0]->info().width()),
        _dest_height(_outputs[0]->info().height_batch())
{
    _crop_param = std::make_shared<RaliRandomCropParam>(_batch_size);
}

void RandomCropNode::create_node()
{
    if(_node)
        return;

    if(_dest_width == 0 || _dest_height == 0)
        THROW("Uninitialized destination dimension")

    _crop_param->create_array(_graph);
    _node = vxExtrppNode_CropPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _crop_param->cropw_arr,
                                _crop_param->croph_arr, _crop_param->x1_arr, _crop_param->y1_arr, _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Error adding the crop resize node (vxExtrppNode_ResizeCropbatchPD    ) failed: "+TOSTR(status))
}

void RandomCropNode::update_node()
{
    _crop_param->set_image_dimensions(_inputs[0]->info().get_roi_width_vec(), _inputs[0]->info().get_roi_height_vec());
    _crop_param->update_array();
    std::vector<uint32_t> crop_h_dims, crop_w_dims;
    _crop_param->get_crop_dimensions(crop_w_dims, crop_h_dims);
    _outputs[0]->update_image_roi(crop_w_dims, crop_h_dims);  
}

void RandomCropNode::init(float crop_area_factor, float crop_aspect_ratio, float x_drift, float y_drift)
{
   
}
void RandomCropNode::init(FloatParam *crop_area_factor, FloatParam *crop_aspect_ratio, FloatParam *x_drift, FloatParam *y_drift, int num_of_attempts)
{
    _crop_param->set_x_drift_factor(core(x_drift));
    _crop_param->set_y_drift_factor(core(y_drift));
    _crop_param->set_area_factor(core(crop_area_factor));
    _crop_param->set_aspect_ratio(core(crop_aspect_ratio));
    _num_of_attempts = num_of_attempts;
}

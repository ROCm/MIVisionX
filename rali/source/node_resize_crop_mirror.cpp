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
#include "node_resize_crop_mirror.h"
#include "exception.h"

ResizeCropMirrorNode::ResizeCropMirrorNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
         Node(inputs, outputs),
        _mirror(MIRROR_RANGE[0], MIRROR_RANGE[1])
{
    _crop_param = std::make_shared<RaliCropParam>(_batch_size);
}

void ResizeCropMirrorNode::create_node()
{
    if(_node)
        return;

    _crop_param->create_array(_graph);

    std::vector<uint32_t> dst_roi_width(_batch_size,_outputs[0]->info().width());
    std::vector<uint32_t> dst_roi_height(_batch_size, _outputs[0]->info().height_single());

    _dst_roi_width = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _dst_roi_height = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);

    vx_status width_status, height_status;

    width_status = vxAddArrayItems(_dst_roi_width, _batch_size, dst_roi_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(_dst_roi_height, _batch_size, dst_roi_height.data(), sizeof(vx_uint32));
    if(width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the crop resize node (vxExtrppNode_ResizeCropbatchPD    )  node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))
    _mirror.create_array(_graph ,VX_TYPE_UINT32, _batch_size);
    _node = vxExtrppNode_ResizeCropMirrorPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _dst_roi_width,
                                           _dst_roi_height, _crop_param->x1_arr, _crop_param->x2_arr, _crop_param->y1_arr, _crop_param->y2_arr,  _mirror.default_array(),_batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Error adding the resize crop mirror resize node (vxExtrppNode_ResizeCropbatchPD    ) failed: "+TOSTR(status))
}

void ResizeCropMirrorNode::update_node()
{
    _crop_param->set_image_dimensions(_inputs[0]->info().get_roi_width_vec(), _inputs[0]->info().get_roi_height_vec());
    _crop_param->update_array();
}

void ResizeCropMirrorNode::init(unsigned int crop_h, unsigned int crop_w, IntParam *mirror)
{
    _crop_param->crop_w = crop_w;
    _crop_param->crop_h = crop_h;
    _crop_param->x1     = 0; 
    _crop_param->y1     = 0;
    _mirror.set_param(core(mirror));
}


void ResizeCropMirrorNode::init(FloatParam *crop_h_factor, FloatParam  *crop_w_factor, IntParam *mirror)
{
    _crop_param->set_crop_height_factor(core(crop_h_factor));
    _crop_param->set_crop_width_factor(core(crop_w_factor));
    _crop_param->set_random();   
    _mirror.set_param(core(mirror));
}


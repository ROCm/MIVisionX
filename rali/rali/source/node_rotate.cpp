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
#include "node_rotate.h"
#include "exception.h"


RotateNode::RotateNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _angle(ROTATE_ANGLE_RANGE[0], ROTATE_ANGLE_RANGE[1])
{
}

void RotateNode::create_node()
{
    if(_node)
        return;
    std::vector<uint32_t> dst_roi_width(_batch_size,_outputs[0]->info().width());
    std::vector<uint32_t> dst_roi_height(_batch_size, _outputs[0]->info().height_single());

    _dst_roi_width = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _dst_roi_height = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);

    vx_status width_status, height_status;

    width_status = vxAddArrayItems(_dst_roi_width, _batch_size, dst_roi_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(_dst_roi_height, _batch_size, dst_roi_height.data(), sizeof(vx_uint32));
    if(width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the resize (vxExtrppNode_ResizebatchPD) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))

    _angle.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
   _node = vxExtrppNode_RotatebatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _dst_roi_width, _dst_roi_height, _angle.default_array(), _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the rotate (vxExtrppNode_RotatebatchPD) node failed: "+ TOSTR(status))

}

void RotateNode::init(float angle)
{
    _angle.set_param(angle);
}

void RotateNode::init(FloatParam* angle)
{
    _angle.set_param(core(angle));
}

void RotateNode::update_node()
{
    _angle.update_array();
}

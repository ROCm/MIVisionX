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
#include "node_contrast.h"
#include "exception.h"

RaliContrastNode::RaliContrastNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _min(CONTRAST_MIN_RANGE[0], CONTRAST_MIN_RANGE[1]),
        _max(CONTRAST_MAX_RANGE[0], CONTRAST_MAX_RANGE[1])
{
}

void RaliContrastNode::create_node()
{

    if(_node)
        return;

    _min.create_array(_graph ,VX_TYPE_UINT32, _batch_size);
    _max.create_array(_graph ,VX_TYPE_UINT32 , _batch_size);

    _node = vxExtrppNode_ContrastbatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _min.default_array(), _max.default_array(), _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the contrast (vxExtrppNode_contrast) node failed: "+ TOSTR(status))
}

void RaliContrastNode::init(int min, int max)
{
    _min.set_param(min);
    _max.set_param(max);
}

void RaliContrastNode::init(IntParam *min, IntParam* max)
{
    _min.set_param(core(min));
    _max.set_param(core(max));
}

void RaliContrastNode::update_node()
{
    _min.update_array();
    _max.update_array();
}


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

#include "node.h"
Node::~Node()
{

    if(!_node)
        vxReleaseNode(&_node);
    _node = nullptr;
}

void
Node::create(std::shared_ptr<Graph> graph)
{
    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output images to the node")

    _graph = graph;

    if(!_inputs.empty())
    {
        vx_status width_status, height_status;
        std::vector<uint32_t> roi_width(_batch_size, _inputs[0]->info().width());
        std::vector<uint32_t> roi_height(_batch_size, _inputs[0]->info().height_single());
        _src_roi_width = vxCreateArray(vxGetContext((vx_reference) _graph->get()), VX_TYPE_UINT32, _batch_size);
        _src_roi_height = vxCreateArray(vxGetContext((vx_reference) _graph->get()), VX_TYPE_UINT32, _batch_size);
        width_status = vxAddArrayItems(_src_roi_width, _batch_size, roi_width.data(), sizeof(vx_uint32));
        height_status = vxAddArrayItems(_src_roi_height, _batch_size, roi_height.data(), sizeof(vx_uint32));
        if (width_status != 0 || height_status != 0)
            THROW(" vxAddArrayItems failed : " + TOSTR(width_status) + "  " + TOSTR(height_status))
    }

    create_node();
}

void
Node::update_parameters()
{
    update_src_roi();
    update_node();
}

void
Node::update_src_roi()
{
    vx_status width_status, height_status;
    width_status = vxCopyArrayRange((vx_array)_src_roi_width, 0, _batch_size, sizeof(vx_uint32), _inputs[0]->info().get_roi_width(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    height_status = vxCopyArrayRange((vx_array)_src_roi_height, 0, _batch_size, sizeof(vx_uint32), _inputs[0]->info().get_roi_height(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(width_status != 0 || height_status != 0)
        THROW(" Failed calling vxCopyArrayRange for width / height status : "+ TOSTR(width_status) + " / "+ TOSTR(height_status))
}
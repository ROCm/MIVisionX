/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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
#include "node_resize.h"
#include "exception.h"


ResizeNode::ResizeNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs)
{
}

void ResizeNode::create_node()
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

   _node = vxExtrppNode_ResizebatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _dst_roi_width, _dst_roi_height, _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the resize (vxExtrppNode_ResizebatchPD) node failed: "+ TOSTR(status))
}

void ResizeNode::update_node() {
    std::vector<uint32_t> src_h_dims, src_w_dims;
    src_w_dims = _inputs[0]->info().get_roi_width_vec();
    src_h_dims = _inputs[0]->info().get_roi_height_vec();
    for (unsigned i = 0; i < _batch_size; i++) {
        _src_roi_size[0] = src_w_dims[i];
        _src_roi_size[1] = src_h_dims[i];
        std::cerr << "\n _src_roi_size[0] :" << _src_roi_size[0] << "  _src_roi_size[1] : " << _src_roi_size[1] << std::endl; 
        _dst_roi_size[0] = _dest_width;
        _dst_roi_size[1] = _dest_height;
        std::cerr << "\n _dst_roi_size[0] :" << _dst_roi_size[0] << "  _dst_roi_size[1] : " << _dst_roi_size[1] << std::endl; 
        adjust_out_roi_size();
        std::cerr << "\n Dest width & height  : " << _dst_roi_size[0] << " x "<< _dst_roi_size[1] << std::endl;
        _dst_roi_size[0] = std::min(_dst_roi_size[0], _outputs[0]->info().width());
        _dst_roi_size[1] = std::min(_dst_roi_size[1], _outputs[0]->info().height_single());
        _dst_roi_width_vec.push_back(_dst_roi_size[0]);
        _dst_roi_height_vec.push_back(_dst_roi_size[1]);
    }
    vx_status width_status, height_status;
    width_status = vxCopyArrayRange((vx_array)_dst_roi_width, 0, _batch_size, sizeof(vx_uint32), _dst_roi_width_vec.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    height_status = vxCopyArrayRange((vx_array)_dst_roi_height, 0, _batch_size, sizeof(vx_uint32), _dst_roi_height_vec.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(width_status != 0 || height_status != 0)
        WRN("ERROR: vxCopyArrayRange _dst_roi_width or _dst_roi_height failed " + TOSTR(width_status) + "  " + TOSTR(height_status));
    _outputs[0]->update_image_roi(_dst_roi_width_vec, _dst_roi_height_vec);
    _dst_roi_width_vec.clear();
    _dst_roi_height_vec.clear();
}

void ResizeNode::init(unsigned dest_width, unsigned dest_height, RocalResizeScalingMode scaling_mode,
                      std::vector<unsigned> max_size, RocalResizeInterpolationType interpolation_type) {
    _scaling_mode = scaling_mode;
    _dest_width = dest_width;
    _dest_height = dest_height;
    _interpolation_type = (int)interpolation_type;
    _src_roi_size.resize(2);
    _dst_roi_size.resize(2);
    _max_roi_size = max_size;
}

void ResizeNode::adjust_out_roi_size() {
    unsigned int dst_w = _dst_roi_size[0];
    unsigned int dst_h = _dst_roi_size[1];
    bool has_max_size = _max_roi_size.size() > 0;
    if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_STRETCH)
    {
        if(_dst_roi_size[0] == 0 and _dst_roi_size[1] != 0)
            _dst_roi_size[0] = _src_roi_size[0];
        else if (_dst_roi_size[1] == 0 and _dst_roi_size[0] != 0)
            _dst_roi_size[1] = _src_roi_size[1];
        if (has_max_size)
        {
            for (unsigned i = 0; i < 2; i++)
            {
                if ((_max_roi_size[i] > 0) && (_dst_roi_size[i] > _max_roi_size[i]))
                    _dst_roi_size[i] = _max_roi_size[i];
            }
        }
    } 
    else if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_DEFAULT)
    {
        float scale;
        if(_dst_roi_size[0] == 0 and _dst_roi_size[1] != 0)
        {
            scale = static_cast<float>(_dst_roi_size[1]) / static_cast<float>(_src_roi_size[1]);
            _dst_roi_size[0] = static_cast<int>(std::round(_src_roi_size[0] * scale));
        }
        else if(_dst_roi_size[1] == 0 and _dst_roi_size[0] != 0)
        {
            scale = static_cast<float>(_dst_roi_size[0]) / static_cast<float>(_src_roi_size[0]);
            _dst_roi_size[1] = static_cast<int>(std::round(_src_roi_size[1] * scale));
        }
        if (has_max_size)
        {
            for (unsigned i = 0; i < 2; i++)
            {
                if ((_max_roi_size[i] > 0) && (_dst_roi_size[i] > _max_roi_size[i]))
                    _dst_roi_size[i] = _max_roi_size[i];
            }
        }
    }
    else
    {
        //if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_SMALLER)
        dst_w = _dst_roi_size[0];
        dst_h = _dst_roi_size[1];
        float scale_w, scale_h, scale;
        if (_dst_roi_size[0] != 0 && _dst_roi_size[1] != 0)
        {
                scale_w = static_cast<float>(_dst_roi_size[0]) / static_cast<float>(_src_roi_size[0]);
                scale_h = static_cast<float>(_dst_roi_size[1]) / static_cast<float>(_src_roi_size[1]);
                if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_SMALLER)
                    scale = std::max(scale_w, scale_h);
                else if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_LARGER)
                    scale = std::min(scale_w, scale_h);
                if (scale_w != scale) // W > H
                    dst_w = static_cast<int>(std::round(_src_roi_size[0] * scale));
                if (scale_h != scale) // H > W
                    dst_h = static_cast<int>(std::round(_src_roi_size[1] * scale));
        }
        else if(_dst_roi_size[0] == 0 and _dst_roi_size[1] != 0)
        {
            scale = scale_h = static_cast<float>(_dst_roi_size[1]) / static_cast<float>(_src_roi_size[1]);
            dst_w = static_cast<int>(std::round(_src_roi_size[0] * scale));
        }
        else if(_dst_roi_size[1] == 0 and _dst_roi_size[0] != 0)
        {
            scale = scale_w = static_cast<float>(_dst_roi_size[0]) / static_cast<float>(_src_roi_size[0]);
            dst_h= static_cast<int>(std::round(_src_roi_size[1] * scale));
        }
        if (has_max_size)
        {
            if ((_max_roi_size[1] > 0) && (dst_h > _max_roi_size[1]))
            {
                dst_h = _max_roi_size[1];
                scale = static_cast<float>(_max_roi_size[1]) / static_cast<float>(_src_roi_size[1]);
                dst_w = std::round(_src_roi_size[0] * scale);
            }
            if ((_max_roi_size[0] > 0) && (dst_w > _max_roi_size[0]))
            {
                dst_w = _max_roi_size[0];
                scale = static_cast<float>(_max_roi_size[0]) / static_cast<float>(_src_roi_size[0]);
                dst_h = std::round(_src_roi_size[1] * scale);
            }
        }
    }
    _dst_roi_size[0] = dst_w;
    _dst_roi_size[1] = dst_h;
}

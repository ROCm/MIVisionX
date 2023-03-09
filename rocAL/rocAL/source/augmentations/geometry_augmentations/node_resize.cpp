/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

void ResizeNode::create_node() {
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
#if ENABLE_OPENCL
    // fall back to batch_pd version since there is no tensor implementation for OPENCL    
    _node = vxExtrppNode_ResizebatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _dst_roi_width, _dst_roi_height, _batch_size);
#else
   _node = vxExtrppNode_Resizetensor(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _dst_roi_width, _dst_roi_height, _interpolation_type, _batch_size);
#endif
    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the resize (vxExtrppNode_Resizetensor) node failed: "+ TOSTR(status))
}

void ResizeNode::update_node() {
    std::vector<uint32_t> src_h_dims, src_w_dims;
    // Using original width and height instead of the decoded width and height for computing resize dimensions
    src_w_dims = _inputs[0]->info().get_original_width_vec();
    src_h_dims = _inputs[0]->info().get_original_height_vec();
    for (unsigned i = 0; i < _batch_size; i++) {
        _src_width = src_w_dims[i];
        _src_height = src_h_dims[i];
        _dst_width = _out_width;
        _dst_height = _out_height;
        adjust_out_roi_size();
        _dst_width = std::min(_dst_width, _outputs[0]->info().width());
        _dst_height = std::min(_dst_height, _outputs[0]->info().height_single());
        _dst_roi_width_vec.push_back(_dst_width);
        _dst_roi_height_vec.push_back(_dst_height);
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
                      const std::vector<unsigned>& max_size, RocalResizeInterpolationType interpolation_type) {
    _interpolation_type = (int)interpolation_type;
    _scaling_mode = scaling_mode;
    _out_width = dest_width;
    _out_height = dest_height;
    if(max_size.size() > 0) {
        _max_width = max_size[0];
        _max_height = max_size[1];
    }
}

void ResizeNode::adjust_out_roi_size() {
    bool has_max_size = (_max_width | _max_height) > 0;

    if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_STRETCH) {
        if (!_dst_width) _dst_width = _src_width;
        if (!_dst_height) _dst_height = _src_height;

        if (has_max_size) {
            if (_max_width) _dst_width = std::min(_dst_width, _max_width);
            if (_max_height) _dst_height = std::min(_dst_height, _max_height);
        }
    } else if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_DEFAULT) {
        if ((!_dst_width) & _dst_height) {  // Only height is passed
            _dst_width = std::lround(_src_width * (static_cast<float>(_dst_height) / _src_height));
        } else if ((!_dst_height) & _dst_width) {  // Only width is passed
            _dst_height = std::lround(_src_height * (static_cast<float>(_dst_width) / _src_width));
        }
        
        if (has_max_size) {
            if (_max_width) _dst_width = std::min(_dst_width, _max_width);
            if (_max_height) _dst_height = std::min(_dst_height, _max_height);
        }
    } else {
        float scale = 1.0f;
        float scale_w = static_cast<float>(_dst_width) / _src_width;
        float scale_h = static_cast<float>(_dst_height) / _src_height;
        if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_SMALLER) {
            scale = std::max(scale_w, scale_h);
        } else if (_scaling_mode == RocalResizeScalingMode::ROCAL_SCALING_MODE_NOT_LARGER) {
            scale = (scale_w > 0 && scale_h > 0) ? std::min(scale_w, scale_h) : ((scale_w > 0) ? scale_w : scale_h);
        }
        
        if (has_max_size) {
            if (_max_width) scale = std::min(scale, static_cast<float>(_max_width) / _src_width);
            if (_max_height) scale = std::min(scale, static_cast<float>(_max_height) / _src_height);
        }

        if ((scale_h != scale) || (!_dst_height)) _dst_height = std::lround(_src_height * scale);
        if ((scale_w != scale) || (!_dst_width)) _dst_width = std::lround(_src_width * scale);
    }
}

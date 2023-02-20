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
#include <cmath>
#include "node_resize_mirror_normalize.h"
#include "exception.h"

ResizeMirrorNormalizeNode::ResizeMirrorNormalizeNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs), _mirror(MIRROR_RANGE[0], MIRROR_RANGE[1])
{
}

void ResizeMirrorNormalizeNode::create_node()
{
    if(_node)
        return;

    std::vector<vx_float32> mean_vx, std_dev_vx;
    _dest_width_val.resize(_batch_size);
    _dest_height_val.resize(_batch_size);
    mean_vx.resize(_batch_size * 3);
    std_dev_vx.resize(_batch_size * 3);
    for (uint i = 0; i < _batch_size; i++) {
        mean_vx[3 * i] = _mean[0];
        mean_vx[3 * i + 1] = _mean[1];
        mean_vx[3 * i + 2] = _mean[2];

        std_dev_vx[3 * i] = _std_dev[0];
        std_dev_vx[3 * i + 1] = _std_dev[1];
        std_dev_vx[3 * i + 2] = _std_dev[2];
    }
    _mean_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size * 3);
    _std_dev_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size * 3);
    vx_status mean_status = VX_SUCCESS;
    mean_status |= vxAddArrayItems(_mean_array,_batch_size * 3, mean_vx.data(), sizeof(vx_float32));
    mean_status |= vxAddArrayItems(_std_dev_array,_batch_size * 3, std_dev_vx.data(), sizeof(vx_float32));
    _mirror.create_array(_graph ,VX_TYPE_UINT32, _batch_size);
    if(mean_status != 0)
        THROW(" vxAddArrayItems failed in the resize mirror normalize node (vxExtrppNode_ResizeMirrorNormalizeCropbatchPD    )  node: "+ TOSTR(mean_status) + "  "+ TOSTR(mean_status))
    
    unsigned int chnShift = 0;
    vx_scalar chnToggle = vxCreateScalar(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, &chnShift);

    std::vector<uint32_t> dst_roi_width(_batch_size,_outputs[0]->info().width());
    std::vector<uint32_t> dst_roi_height(_batch_size, _outputs[0]->info().height_single());

    _dst_roi_width = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _dst_roi_height = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    vx_status width_status, height_status;

    width_status = vxAddArrayItems(_dst_roi_width, _batch_size, dst_roi_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(_dst_roi_height, _batch_size, dst_roi_height.data(), sizeof(vx_uint32));
     if(width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the resize mirror normalize node (vxExtrppNode_ResizeMirrorNormalizeCropbatchPD ) node: " + TOSTR(width_status) + "  " + TOSTR(height_status))

   _node = vxExtrppNode_ResizeMirrorNormalizeTensor(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(),
                                                    _dst_roi_width, _dst_roi_height, _mean_array, _std_dev_array,
                                                    _mirror.default_array(), chnToggle, _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the resize mirror normalize node (vxExtrppNode_ResizeMirrorNormalizeCropbatchPD) node failed: "+ TOSTR(status))
}

void ResizeMirrorNormalizeNode::update_node()
{
    std::vector<uint32_t> src_roi_width, src_roi_height;
    src_roi_width = _inputs[0]->info().get_roi_width_vec();
    src_roi_height = _inputs[0]->info().get_roi_height_vec();

    for(uint i = 0; i < _batch_size; i++)
    {
        // Min size and max size used for MLPerf MaskRCNN resize augmentation
        // TODO: Get the min_size and max_size as user arguments from python
        int min_size = 800;
        int max_size = 1333;
        int src_width = src_roi_width[i];
        int src_height = src_roi_height[i];
        int size = min_size;
        int output_width, output_height;

        float min_original_size = static_cast<float>(std::min(src_width, src_height));
        float max_original_size = static_cast<float>(std::max(src_width, src_height));
        if(max_original_size / min_original_size * size > max_size)
            size = static_cast<size_t>(round(max_size * min_original_size / max_original_size));

        if (((src_width <= src_height) && (src_width == size)) || ((src_height <= src_width) && (src_height == size)))
        {
            _dest_height_val[i] = src_height;
            _dest_width_val[i] = src_width;
            continue;
        }

        if(src_width < src_height) {
            output_width = size;
            output_height = static_cast<size_t>(size * src_height / src_width);	
        } else {
            output_height = size;
            output_width = static_cast<size_t>(size * src_width / src_height);
        }
	_dest_height_val[i] = output_height;
	_dest_width_val[i] = output_width;
    }
    vxCopyArrayRange((vx_array)_dst_roi_width, 0, _batch_size, sizeof(uint), _dest_width_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_dst_roi_height, 0, _batch_size, sizeof(uint), _dest_height_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

    _outputs[0]->update_image_roi(_dest_width_val, _dest_height_val);
    _mirror.update_array();
}

void ResizeMirrorNormalizeNode::init(std::vector<float>& mean, std::vector<float>& std_dev, IntParam *mirror)
{
    _mean   = mean;
    _std_dev = std_dev;
    _mirror.set_param(core(mirror));
}

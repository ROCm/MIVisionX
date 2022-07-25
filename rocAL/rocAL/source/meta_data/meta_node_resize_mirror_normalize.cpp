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

#include "meta_node_resize_mirror_normalize.h"

void ResizeMirrorNormalizeMetaNode::initialize()
{
    _src_height_val.resize(_batch_size);
    _src_width_val.resize(_batch_size);
    _dst_width_val.resize(_batch_size);
    _dst_height_val.resize(_batch_size);
    _mirror_val.resize(_batch_size);
}
void ResizeMirrorNormalizeMetaNode::update_parameters(MetaDataBatch *input_meta_data, bool segmentation)
{
    initialize();
    if (_batch_size != input_meta_data->size())
    {
        _batch_size = input_meta_data->size();
    }
    _mirror = _node->return_mirror();
    _src_width = _node->get_src_width();
    _src_height = _node->get_src_height();
    _dst_width = _node->get_dst_width();
    _dst_height = _node->get_dst_height();

    vxCopyArrayRange((vx_array)_mirror, 0, _batch_size, sizeof(uint), _mirror_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_src_width, 0, _batch_size, sizeof(uint), _src_width_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_src_height, 0, _batch_size, sizeof(uint), _src_height_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_dst_width, 0, _batch_size, sizeof(uint), _dst_width_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_dst_height, 0, _batch_size, sizeof(uint), _dst_height_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    for (int i = 0; i < _batch_size; i++)
    {
        _dst_to_src_width_ratio = _dst_width_val[i] / float(_src_width_val[i]);
        _dst_to_src_height_ratio = _dst_height_val[i] / float(_src_height_val[i]);
        auto bb_count = input_meta_data->get_bb_labels_batch()[i].size();
        BoundingBoxCords coords_buf;
        BoundingBoxLabels labels_buf;
        coords_buf.resize(bb_count);
        labels_buf.resize(bb_count);
        memcpy(labels_buf.data(), input_meta_data->get_bb_labels_batch()[i].data(), sizeof(int) * bb_count);
        memcpy((void *)coords_buf.data(), input_meta_data->get_bb_cords_batch()[i].data(), input_meta_data->get_bb_cords_batch()[i].size() * sizeof(BoundingBoxCord));
        BoundingBoxCords bb_coords;
        BoundingBoxLabels bb_labels;
        if (segmentation)
        {
            // auto ptr = mask_data;
            auto mask_data_ptr = input_meta_data->get_mask_cords_batch()[i].data();
            int mask_size = input_meta_data->get_mask_cords_batch()[i].size();
            for (int idx = 0; idx < mask_size; idx += 2)
            {
                if(_mirror_val[i] == 1)
                {
                    mask_data_ptr[idx] = _dst_width_val[i] - (mask_data_ptr[idx] * _dst_to_src_width_ratio) - 1;
                    mask_data_ptr[idx + 1] = mask_data_ptr[idx + 1] * _dst_to_src_height_ratio;
                }
                else
                {
                    mask_data_ptr[idx] = mask_data_ptr[idx] * _dst_to_src_width_ratio;
                    mask_data_ptr[idx + 1] = mask_data_ptr[idx + 1] * _dst_to_src_height_ratio;
                }
            }
        }

        for (uint j = 0; j < bb_count; j++)
        {            
            if(_mirror_val[i] == 1)
            {
                float one_by_width_coeff = 1 / float(_dst_width_val[i]);
                float l = 1 - coords_buf[j].r - one_by_width_coeff;
                coords_buf[j].r = 1 - coords_buf[j].l - one_by_width_coeff;
                coords_buf[j].l = l; 
            }
            bb_coords.push_back(coords_buf[j]);
            bb_labels.push_back(labels_buf[j]);
        }
        input_meta_data->get_bb_cords_batch()[i] = bb_coords;
        input_meta_data->get_bb_labels_batch()[i] = bb_labels;
    }
}

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

#include "meta_node_flip.h"
void FlipMetaNode::initialize()
{
    _src_height_val.resize(_batch_size);
    _src_width_val.resize(_batch_size);
    _flip_axis_val.resize(_batch_size);
}
void FlipMetaNode::update_parameters(MetaDataBatch* input_meta_data)
{
    initialize();
    if(_batch_size != input_meta_data->size())
    {
        _batch_size = input_meta_data->size();
    }
    _src_width = _node->get_src_width();
    _src_height = _node->get_src_height();
    _flip_axis = _node->get_flip_axis();
    vxCopyArrayRange((vx_array)_src_width, 0, _batch_size, sizeof(uint),_src_width_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_src_height, 0, _batch_size, sizeof(uint),_src_height_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_flip_axis, 0, _batch_size, sizeof(int),_flip_axis_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    for(int i = 0; i < _batch_size; i++)
    {
        auto bb_count = input_meta_data->get_bb_labels_batch()[i].size();
        BoundingBoxLabels labels_buf;
        BoundingBoxCords coords_buf;
        coords_buf.resize(bb_count);
        labels_buf.resize(bb_count);
        memcpy(labels_buf.data(), input_meta_data->get_bb_labels_batch()[i].data(),  sizeof(int)*bb_count);
        memcpy((void *)coords_buf.data(), input_meta_data->get_bb_cords_batch()[i].data(), input_meta_data->get_bb_cords_batch()[i].size() * sizeof(BoundingBoxCord));
        BoundingBoxCords bb_coords;
        
        for(uint j = 0; j < bb_count; j++)
        {
            if(_flip_axis_val[i] == 0)
            {
                float l = 1 - coords_buf[j].r;
                coords_buf[j].r = 1 - coords_buf[j].l;
                coords_buf[j].l = l;     
            }
            else if(_flip_axis_val[i] == 1)
            {
                float t = 1 - coords_buf[j].b;
                coords_buf[j].b = 1 - coords_buf[j].t;
                coords_buf[j].t = t;
            }
            
            bb_coords.push_back(coords_buf[j]);
        }
        input_meta_data->get_bb_cords_batch()[i] = bb_coords;
        input_meta_data->get_bb_labels_batch()[i] = labels_buf;
    }
}

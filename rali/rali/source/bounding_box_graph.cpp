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
#include "bounding_box_graph.h"

void BoundingBoxGraph::process(MetaDataBatch *meta_data)
{
    for (auto &meta_node : _meta_nodes)
    {
        meta_node->update_parameters(meta_data);
    }
}

void BoundingBoxGraph::update_meta_data(MetaDataBatch *input_meta_data, decoded_image_info decode_image_info)
{
    std::vector<uint32_t> original_height = decode_image_info._original_height;
    std::vector<uint32_t> original_width = decode_image_info._original_width;
    std::vector<uint32_t> roi_width = decode_image_info._roi_width;
    std::vector<uint32_t> roi_height = decode_image_info._roi_height;
    for (int i = 0; i < input_meta_data->size(); i++)
    {
        float _dst_to_src_width_ratio = roi_width[i] / float(original_width[i]);
        float _dst_to_src_height_ratio = roi_height[i] / float(original_height[i]);
        unsigned bb_count = input_meta_data->get_bb_labels_batch()[i].size();
        std::vector<int> labels_buf(bb_count);
        std::vector<float> coords_buf(bb_count * 4);
        memcpy(labels_buf.data(), input_meta_data->get_bb_labels_batch()[i].data(), sizeof(int) * bb_count);
        memcpy(coords_buf.data(), input_meta_data->get_bb_cords_batch()[i].data(), input_meta_data->get_bb_cords_batch()[i].size() * sizeof(BoundingBoxCord));
        BoundingBoxCords bb_coords;
        BoundingBoxCord temp_box;
        temp_box.x = temp_box.y = temp_box.w = temp_box.h = 0;
        BoundingBoxLabels bb_labels;
        int m = 0;
        for (uint j = 0; j < bb_count; j++)
        {
            BoundingBoxCord box;
            float temp_x, temp_y;
            temp_x = (coords_buf[m++] * _dst_to_src_width_ratio);
            temp_y = (coords_buf[m++] * _dst_to_src_height_ratio);
            box.x = (temp_x > 0) ? temp_x : 0;
            box.y = (temp_y > 0) ? temp_y : 0;
            box.w = (coords_buf[m++] * _dst_to_src_width_ratio);
            box.h = (coords_buf[m++] * _dst_to_src_height_ratio);
            bb_coords.push_back(box);
            bb_labels.push_back(labels_buf[j]);
        }
        if (bb_coords.size() == 0)
        {
            bb_coords.push_back(temp_box);
            bb_labels.push_back(0);
        }
        input_meta_data->get_bb_cords_batch()[i] = bb_coords;
        input_meta_data->get_bb_labels_batch()[i] = bb_labels;
    }
}

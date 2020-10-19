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

#include "meta_node_rotate.h"
void RotateMetaNode::initialize()
{
    _src_height_val.resize(_batch_size);
    _src_width_val.resize(_batch_size);
    _angle_val.resize(_batch_size);
}
float calculate_rotate_point_x(float angle, float x, float cx, float cy)
{
    float rotate[2];
    rotate[0] = cos(RAD(angle));
    rotate[1] = -1 * sin(RAD(angle));
    float trans_x = cx + (rotate[0] * cx) + (rotate[1] * cy);
    return trans_x;
}
float calculate_rotate_point_y(float angle, float y, float cx, float cy)
{
    float rotate[2];
    rotate[0] = sin(RAD(angle));
    rotate[1] = cos(RAD(angle));
    float trans_y = cy + (rotate[0] * cx) + (rotate[1] * cy);
    return trans_y;
}
void RotateMetaNode::update_parameters(MetaDataBatch* input_meta_data)
{
    initialize();
    if(_batch_size != input_meta_data->size())
    {
        _batch_size = input_meta_data->size();
    }
    _src_width = _node->get_src_width();
    _src_height = _node->get_src_height();
    _dst_width = _node->get_dst_width();
    _dst_height = _node->get_dst_height();
    _angle = _node->get_angle();
    vxCopyArrayRange((vx_array)_src_width, 0, _batch_size, sizeof(uint),_src_width_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_src_height, 0, _batch_size, sizeof(uint),_src_height_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_angle, 0, _batch_size, sizeof(float),_angle_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    for(int i = 0; i < _batch_size; i++)
    {
        auto bb_count = input_meta_data->get_bb_labels_batch()[i].size();
        int labels_buf[bb_count];
        float coords_buf[bb_count*4];
        memcpy(labels_buf, input_meta_data->get_bb_labels_batch()[i].data(),  sizeof(int)*bb_count);
        memcpy(coords_buf, input_meta_data->get_bb_cords_batch()[i].data(), input_meta_data->get_bb_cords_batch()[i].size() * sizeof(BoundingBoxCord));
        BoundingBoxCords bb_coords;
        BoundingBoxCord temp_box;
        temp_box.x = temp_box.y = temp_box.w = temp_box.h = 0;
        BoundingBoxLabels bb_labels;
        BoundingBoxCord dest_image;
        dest_image.x = 0;
        dest_image.y = 0;
        dest_image.w = _dst_width;
        dest_image.h = _dst_height;
        for(uint j = 0, m = 0; j < bb_count; j++)
        {
            BoundingBoxCord box;
            float src_bb_x, src_bb_y, bb_w, bb_h, dest_bb_x, dest_bb_y;
            float cx, cy;
            float x1, y1, x2, y2, x3, y3, x4, y4, min_x, min_y, max_x, max_y;
            cx = _dst_width / 2;
            cy = _dst_height / 2;
            src_bb_x = (coords_buf[m++]);
            src_bb_y = (coords_buf[m++]);
            bb_w = (coords_buf[m++]);
            bb_h = (coords_buf[m++]);
            dest_bb_x = (cx) - ((_src_width_val[i] / 2) - src_bb_x);
            dest_bb_y = (cy) - ((_src_height_val[i] / 2) - src_bb_y);
            x1 = calculate_rotate_point_x(dest_bb_x, _angle_val[i], cx, cy);
            y1 = calculate_rotate_point_y(dest_bb_y, _angle_val[i], cx, cy);
            x2 = calculate_rotate_point_x(dest_bb_x + bb_w, _angle_val[i], cx, cy);
            y2 = calculate_rotate_point_y(dest_bb_y, _angle_val[i], cx, cy);
            x3 = calculate_rotate_point_x(dest_bb_x, _angle_val[i], cx, cy);
            y3 = calculate_rotate_point_y(dest_bb_y + bb_h, _angle_val[i], cx, cy);
            x4 = calculate_rotate_point_x(dest_bb_x + bb_w, _angle_val[i], cx, cy);
            y4 = calculate_rotate_point_y(dest_bb_y + bb_h, _angle_val[i], cx, cy);
            min_x = std::min(x1, std::min(x2, std::min(x3, x4)));
            min_y = std::min(y1, std::min(y2, std::min(y3, y4)));
            max_x = std::max(x1, std::max(x2, std::max(x3, x4)));
            max_y = std::max(y1, std::max(y2, std::max(y3, y4)));
            box.x = (min_x > 0) ? min_x : 0;
            box.y = (min_y > 0) ? min_y : 0;
            box.w = max_x - min_x;
            box.h = max_y - min_y;
            if(((box.x + box.w) <= dest_image.w) ? ((box.y + box.h) <= dest_image.h) ? true : false : false)
            {
                    bb_coords.push_back(box);
                    bb_labels.push_back(labels_buf[j]);
            }
            else
            {
                if (BBoxIntersectionOverUnion(box, dest_image) >= _iou_threshold)
                {
                    if ((box.x + box.w) > dest_image.w)
                        box.w = dest_image.w - box.x;
                    if ((box.y + box.h) > dest_image.h)
                        box.h = dest_image.h - box.y;
                    bb_coords.push_back(box);
                    bb_labels.push_back(labels_buf[j]);
                }
            }
        }
        if(bb_coords.size() == 0)
        {
            bb_coords.push_back(temp_box);
            bb_labels.push_back(0);
        }
        input_meta_data->get_bb_cords_batch()[i] = bb_coords;
        input_meta_data->get_bb_labels_batch()[i] = bb_labels;
    }
}

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


#include "meta_node_crop_mirror_normalize.h"
void CropMirrorNormalizeMetaNode::initialize()
{
    _width_val.resize(_batch_size);
    _height_val.resize(_batch_size);
    _x1_val.resize(_batch_size);
    _y1_val.resize(_batch_size);
    _mirror_val.resize(_batch_size);
}
void CropMirrorNormalizeMetaNode::update_parameters(MetaDataBatch* input_meta_data)
{
    initialize();
    if(_batch_size != input_meta_data->size())
    {
        _batch_size = input_meta_data->size();
    }
    _mirror = _node->return_mirror();
    _meta_crop_param = _node->return_crop_param();
    _dstImgWidth = _meta_crop_param->cropw_arr;
    _dstImgHeight = _meta_crop_param->croph_arr;
    _x1 = _meta_crop_param->x1_arr;
    _y1 = _meta_crop_param->y1_arr;
    vxCopyArrayRange((vx_array)_dstImgWidth, 0, _batch_size, sizeof(uint),_width_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_dstImgHeight, 0, _batch_size, sizeof(uint),_height_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_x1, 0, _batch_size, sizeof(uint),_x1_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_y1, 0, _batch_size, sizeof(uint),_y1_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_mirror, 0, _batch_size, sizeof(uint),_mirror_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    for(int i = 0; i < _batch_size; i++)
    {
        auto bb_count = input_meta_data->get_bb_labels_batch()[i].size();
        int labels_buf[bb_count];
        float coords_buf[bb_count*4];
        memcpy(labels_buf, input_meta_data->get_bb_labels_batch()[i].data(),  sizeof(int)*bb_count);
        memcpy(coords_buf, input_meta_data->get_bb_cords_batch()[i].data(), input_meta_data->get_bb_cords_batch()[i].size() * sizeof(BoundingBoxCord));
        BoundingBoxCords bb_coords;
        BoundingBoxCord temp_box;
        BoundingBoxLabels bb_labels;
        BoundingBoxCord crop_box;
        crop_box.x = _x1_val[i];
        crop_box.y = _y1_val[i];
        crop_box.w = _width_val[i];
        crop_box.h = _height_val[i];
        for(uint j = 0, m = 0; j < bb_count; j++)
        {
            BoundingBoxCord box;
            box.x = coords_buf[m++];
            box.y = coords_buf[m++];
            box.w = coords_buf[m++];
            box.h = coords_buf[m++];
            if (BBoxIntersectionOverUnion(box, crop_box) >= _iou_threshold)
            {
                float xA = std::max(crop_box.x, box.x);
                float yA = std::max(crop_box.y, box.y);
                float xB = std::min(crop_box.x + crop_box.w, box.x + box.w);
                float yB = std::min(crop_box.y + crop_box.h, box.y + box.h);
                box.x = xA - _x1_val[i];
                box.y = yA - _y1_val[i];
                box.w = xB - xA;
                box.h = yB - yA;
                if(_mirror_val[i] == 1)
                {
                    float centre_x = _width_val[i] / 2;
                    box.x += ((centre_x - box.x) * 2) - box.w;
                }
                bb_coords.push_back(box);
                bb_labels.push_back(labels_buf[j]);
            }
        }
        if(bb_coords.size() == 0)
        {
	        //std::cout << "Crop mirror Normalize - Zero Bounding boxes" << std::endl;
            temp_box.x = 0;
            temp_box.y = 0;
	        temp_box.w =  crop_box.w - 1;
	        temp_box.h =  crop_box.h - 1;
            bb_coords.push_back(temp_box);
            bb_labels.push_back(0);
        }
        input_meta_data->get_bb_cords_batch()[i] = bb_coords;
        input_meta_data->get_bb_labels_batch()[i] = bb_labels;
    }
}

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

#include "meta_node_ssd_random_crop.h"
void SSDRandomCropMetaNode::initialize()
{
    _crop_width_val.resize(_batch_size);
    _crop_height_val.resize(_batch_size);
    _x1_val.resize(_batch_size);
    _y1_val.resize(_batch_size);
    _x2_val.resize(_batch_size);
    _y2_val.resize(_batch_size);
}
void SSDRandomCropMetaNode::update_parameters(MetaDataBatch *input_meta_data)
{
    initialize();
    if (_batch_size != input_meta_data->size())
    {
        _batch_size = input_meta_data->size();
    }
    set_threshold(_node->get_threshold());
    _meta_crop_param = _node->get_crop_param();
    _dst_width = _node->get_dst_width();
    _dst_height = _node->get_dst_height();
    _crop_width = _meta_crop_param->cropw_arr;
    _crop_height = _meta_crop_param->croph_arr;
    _x1 = _meta_crop_param->x1_arr;
    _y1 = _meta_crop_param->y1_arr;
    _x2 = _meta_crop_param->x2_arr;
    _y2 = _meta_crop_param->y2_arr;
    vxCopyArrayRange((vx_array)_crop_width, 0, _batch_size, sizeof(uint), _crop_width_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_crop_height, 0, _batch_size, sizeof(uint), _crop_height_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_x1, 0, _batch_size, sizeof(uint), _x1_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_y1, 0, _batch_size, sizeof(uint), _y1_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_x2, 0, _batch_size, sizeof(uint), _x2_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_y2, 0, _batch_size, sizeof(uint), _y2_val.data(), VX_READ_ONLY, VX_MEMORY_TYPE_HOST);

    area_factor = _meta_crop_param->get_area_factor();
    aspect_ratio_factor = _meta_crop_param->get_aspect_ratio();
    x_drift_factor = _meta_crop_param->get_x_drift_factor();
    y_drift_factor = _meta_crop_param->get_y_drift_factor();

    //These Two Variable are suffincient for now
    in_width = _meta_crop_param->in_width;
    in_height = _meta_crop_param->in_height;
    bool invalid_bboxes = true;
    BoundingBoxCord crop_box, jth_box;
    int bb_count;
    for (int i = 0; i < _batch_size; i++)
    {
        bb_count = input_meta_data->get_bb_labels_batch()[i].size();
        std::vector<int> labels_buf(bb_count);
        memcpy(labels_buf.data(), input_meta_data->get_bb_labels_batch()[i].data(), sizeof(int) * bb_count);
        std::vector<float> coords_buf(bb_count * 4);
        memcpy(coords_buf.data(), input_meta_data->get_bb_cords_batch()[i].data(), input_meta_data->get_bb_cords_batch()[i].size() * sizeof(BoundingBoxCord));
        crop_box.h = _crop_height_val[i];
        crop_box.w = _crop_width_val[i];
        crop_box.x = _x1_val[i];
        crop_box.y = _y1_val[i];
        int count = 1000;
        int min_left = INT32_MAX, min_top = INT32_MAX, max_right = INT32_MIN, max_bottom = INT32_MIN;
        while (--count)
        {
            for (int j = 0; j < _num_of_attempts; j++)
            {
                x_drift_factor->renew();
                float factor = 0.5f;
                auto w_factor = factor + (x_drift_factor->get() * (1 - factor));
                crop_box.w = w_factor * in_width[i];
                y_drift_factor->renew();
                y_drift_factor->renew();
                auto h_factor = factor + (y_drift_factor->get() * (1 - factor));
                crop_box.h = h_factor * in_height[i];
                //aspect ratio check
                if ((crop_box.w / crop_box.h < 0.5) || (crop_box.w / crop_box.h > 2.))
                    continue;
                //std::cout << "h_factor * w_factor attempt number external number" << w_factor * h_factor << " " << j << "  "
                //            << count << std::endl;
                break;
            }
            //Got the crop;
            x_drift_factor->renew();
            x_drift_factor->renew();
            y_drift_factor->renew();
            crop_box.x = static_cast<size_t>(x_drift_factor->get() * (in_width[i] - crop_box.w));
            crop_box.y = static_cast<size_t>(y_drift_factor->get() * (in_height[i] - crop_box.h));
            invalid_bboxes = false;
            for (int j = 0; j < bb_count; j++)
            {
                int m = j * 4;
                jth_box.x = coords_buf[m];
                jth_box.y = coords_buf[m + 1];
                jth_box.w = coords_buf[m + 2];
                jth_box.h = coords_buf[m + 3];
                min_left = std::min(min_left, (int)coords_buf[m]);
                min_top = std::min(min_top, (int)coords_buf[m + 1]);
                max_right = std::max(max_right, (int)(coords_buf[m] + coords_buf[m + 2]));
                max_bottom = std::max(max_bottom, (int)(coords_buf[m + 1] + coords_buf[m + 3]));
                if (BBoxIntersectionOverUnion(jth_box, crop_box, false) < 0.5)
                {
                    invalid_bboxes = true;
                    break;
                }
            }
            min_left = std::max(min_left, 0);
            min_top = std::max(min_top, 0);
            max_right = std::min(max_right, (int)in_width[i]);
            max_bottom = std::min(max_bottom, (int)in_height[i]);
            if (invalid_bboxes)
                continue;
            int valid_bbox_count = 0;
            auto left = crop_box.x, top = crop_box.y, right = crop_box.x + crop_box.w, bottom = crop_box.y + crop_box.h;
            for (int j = 0; j < bb_count; j++)
            {
                int m = j * 4;
                auto x_c = 0.5f * (coords_buf[m] + coords_buf[m + 2]);
                auto y_c = 0.5f * (coords_buf[m + 1] + coords_buf[m + 3]);
                if ((x_c >= left) && (x_c <= right) && (y_c >= top) && (y_c <= bottom))
                    valid_bbox_count++;
            }
            if (valid_bbox_count == 0 && bb_count != 0)
                continue;
            break;
        }               // while loop
        if (count == 0) // Did not get a crop even after 1000 attempts of random crops
        {
            _x1_val[i] = std::min(static_cast<size_t>(min_left), static_cast<size_t>(0.1 * in_width[i]));
            _y1_val[i] = std::min(static_cast<size_t>(min_top), static_cast<size_t>(0.1 * in_height[i]));
            _crop_width_val[i] = std::max(static_cast<size_t>(max_right), static_cast<size_t>(0.8 * in_width[i]));
            _crop_height_val[i] = std::max(static_cast<size_t>(max_bottom), static_cast<size_t>(0.8 * in_height[i]));
            crop_box.x = _x1_val[i];
            crop_box.y = _y1_val[i];
            crop_box.w = _crop_width_val[i];
            crop_box.h = _crop_height_val[i];
        }
        else
        {
            _crop_height_val[i] = crop_box.h;
            _crop_width_val[i] = crop_box.w;
            _x1_val[i] = crop_box.x;
            _y1_val[i] = crop_box.y;
        }
        _x2_val[i] = _x1_val[i] + _crop_width_val[i];
        _y2_val[i] = _y1_val[i] + _crop_height_val[i];
        BoundingBoxCords bb_coords;
        BoundingBoxLabels bb_labels;
        for (int j = 0, m = 0; j < bb_count; j++)
        {
            BoundingBoxCord box;
            box.x = coords_buf[m++];
            box.y = coords_buf[m++];
            box.w = coords_buf[m++];
            box.h = coords_buf[m++];
            if (BBoxIntersectionOverUnion(box, crop_box, false) >= 0.5)
            {
                float xA = std::max(crop_box.x, box.x);
                float yA = std::max(crop_box.y, box.y);
                float xB = std::min(crop_box.x + crop_box.w, box.x + box.w);
                float yB = std::min(crop_box.y + crop_box.h, box.y + box.h);
                box.x = xA - _x1_val[i];
                box.y = yA - _y1_val[i];
                box.w = xB - xA;
                box.h = yB - yA;
                bb_coords.push_back(box);
                bb_labels.push_back(labels_buf[j]);
            }
        }
        //std::cout << "Output boxes:" << bb_coords.size() << std::endl;
        //std::cout << "Input Boxes:" << bb_count << std::endl;
        //std::cout << "Cropping Ratio" << (crop_box.w * crop_box.h) / (in_height[i] * in_width[i]) << std::endl;
        if (bb_coords.size() == 0)
        {
            std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ENOCOUNTED CASE of ZERO BBOX!!!!!" << std::endl;
            exit(-1);
        }
        input_meta_data->get_bb_cords_batch()[i] = bb_coords;
        input_meta_data->get_bb_labels_batch()[i] = bb_labels;
    }
    vxCopyArrayRange((vx_array)_crop_width, 0, _batch_size, sizeof(uint), _crop_width_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_crop_height, 0, _batch_size, sizeof(uint), _crop_height_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_x1, 0, _batch_size, sizeof(uint), _x1_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_y1, 0, _batch_size, sizeof(uint), _y1_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_x2, 0, _batch_size, sizeof(uint), _x2_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_y2, 0, _batch_size, sizeof(uint), _y2_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

    _meta_crop_param->croph_arr = _crop_height;
    _meta_crop_param->cropw_arr = _crop_width;
    _meta_crop_param->x1_arr = _x1;
    _meta_crop_param->y1_arr = _y1;
    _node->set_crop_param(_meta_crop_param);
    _node->update_output_dims();
}

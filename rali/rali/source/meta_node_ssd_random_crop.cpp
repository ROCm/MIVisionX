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
    in_width = _meta_crop_param->in_width;
    in_height = _meta_crop_param->in_height;

    int num_of_attempts = 5;
    int count;
    bool invalid_bboxes = true;
    BoundingBoxCord crop_box, jth_box;
    int bb_count;
    for (int i = 0; i < _batch_size; i++)
    {
        bb_count = input_meta_data->get_bb_labels_batch()[i].size();
        std::vector<float> coords_buf(bb_count * 4);
        memcpy(coords_buf.data(), input_meta_data->get_bb_cords_batch()[i].data(), input_meta_data->get_bb_cords_batch()[i].size() * sizeof(BoundingBoxCord));
        crop_box.h = _crop_height_val[i];
        crop_box.w = _crop_width_val[i];
        crop_box.x = _x1_val[i];
        crop_box.y = _y1_val[i];
        count = num_of_attempts;
        while (--count)
        {
            if(crop_box.x > in_width[i]/2 || crop_box.y > in_height[i]/2 || (crop_box.x + crop_box.w) < in_width[i]/2 || (crop_box.y + crop_box.h) < in_height[i]/2)
            {
                crop_box.x = (in_width[i] - crop_box.w) / 2;
                crop_box.y = (in_height[i] - crop_box.h) / 2;
            }
            
            invalid_bboxes = true;
            for (int j = 0, m = 0; j < bb_count; j++)
            {
                jth_box.x = coords_buf[m++];
                jth_box.y = coords_buf[m++];
                jth_box.w = coords_buf[m++];
                jth_box.h = coords_buf[m++];
                if (BBoxIntersectionOverUnion(jth_box, crop_box, false) >= _threshold)
                {
                    invalid_bboxes = false;
                    break;
                }
            }
            if (!invalid_bboxes || bb_count == 0)
            {
                if(bb_count == 0) invalid_bboxes = false;
                break;
            }
            crop_box = generate_random_crop(i);
        }
        if(invalid_bboxes)
        {
            _x1_val[i] = std::min(static_cast<size_t>(coords_buf[0]), static_cast<size_t>(0.1 * in_width[i]));
            _y1_val[i] = std::min(static_cast<size_t>(coords_buf[1]), static_cast<size_t>(0.1 * in_height[i]));
            _crop_width_val[i]  = std::max(static_cast<size_t>(coords_buf[2]), static_cast<size_t>(0.8 * in_width[i]));
            _crop_height_val[i] = std::max(static_cast<size_t>(coords_buf[3]), static_cast<size_t>(0.8 * in_height[i]));
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
    /*update meta data*/
    for (int i = 0; i < _batch_size; i++)
    {
        bb_count = input_meta_data->get_bb_labels_batch()[i].size();
        int labels_buf[bb_count];
        float coords_buf[bb_count * 4];
        memcpy(labels_buf, input_meta_data->get_bb_labels_batch()[i].data(), sizeof(int) * bb_count);
        memcpy(coords_buf, input_meta_data->get_bb_cords_batch()[i].data(), input_meta_data->get_bb_cords_batch()[i].size() * sizeof(BoundingBoxCord));
        BoundingBoxCords bb_coords;
        // BoundingBoxCord temp_box;
        BoundingBoxLabels bb_labels;
        BoundingBoxCord crop_box;
        crop_box.x = _x1_val[i];
        crop_box.y = _y1_val[i];
        crop_box.w = _crop_width_val[i];
        crop_box.h = _crop_height_val[i];
        for (int j = 0, m = 0; j < bb_count; j++)
        {
            BoundingBoxCord box;
            box.x = coords_buf[m++];
            box.y = coords_buf[m++];
            box.w = coords_buf[m++];
            box.h = coords_buf[m++];
            if (BBoxIntersectionOverUnion(box, crop_box, false) >= _threshold)
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
        // if (bb_coords.size() == 0)
        // {
        //     bb_coords.push_back(temp_box);
        //     bb_labels.push_back(0);
        // }
        input_meta_data->get_bb_cords_batch()[i] = bb_coords;
        input_meta_data->get_bb_labels_batch()[i] = bb_labels;
    }
}

BoundingBoxCord SSDRandomCropMetaNode::generate_random_crop(int img_idx)
{
    float area, crop_aspect_ratio, x_drift, y_drift;
    int num_of_attempts = 1; // Can be changed later conveniently
    BoundingBoxCord crop;
    float target_area, in_ratio;
    auto is_valid_crop = [](uint h, uint w, uint height, uint width) {
        return (h < height && w < width);
    };
    for (int i = 0; i < num_of_attempts; i++)
    {
        area_factor->renew();
        aspect_ratio_factor->renew();
        area = std::max(area_factor->get(), 0.25f);
        crop_aspect_ratio = aspect_ratio_factor->get();

        target_area = area * in_height[img_idx] * in_width[img_idx];
        crop.w = static_cast<size_t>(std::sqrt(target_area * crop_aspect_ratio));
        crop.h = static_cast<size_t>(std::sqrt(target_area * (1 / crop_aspect_ratio)));
        if (is_valid_crop(crop.h, crop.w, in_height[img_idx], in_width[img_idx]))
        {
            x_drift_factor->renew();
            y_drift_factor->renew();
            y_drift_factor->renew();
            x_drift = x_drift_factor->get();
            y_drift = y_drift_factor->get();
            crop.x = static_cast<size_t>(x_drift * (in_width[img_idx] - crop.w));
            crop.y = static_cast<size_t>(y_drift * (in_height[img_idx] - crop.h));
            break;
        }
    }
    // Fallback on Central Crop
    if (!is_valid_crop(crop.h, crop.w, in_height[img_idx], in_width[img_idx]))
    {
        in_ratio = static_cast<float>(in_width[img_idx]) / in_height[img_idx];
        if (in_ratio < ASPECT_RATIO_RANGE[0])
        {
            crop.w = in_width[img_idx];
            crop.h = crop.w / ASPECT_RATIO_RANGE[0];
        }
        else if (in_ratio > ASPECT_RATIO_RANGE[1])
        {
            crop.h = in_height[img_idx];
            crop.w = crop.h * ASPECT_RATIO_RANGE[1];
        }
        else
        {
            crop.h = in_height[img_idx];
            crop.w = in_width[img_idx];
        }
        crop.x = (in_width[img_idx] - crop.w) / 2;
        crop.y = (in_height[img_idx] - crop.h) / 2;
    }
    return crop;
}

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
    set_num_of_attempts(_node->get_num_of_attempts());
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

    x_drift_factor = _meta_crop_param->get_x_drift_factor();
    y_drift_factor = _meta_crop_param->get_y_drift_factor();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 6);
    const std::vector<std::pair<float, float>> IOU = {std::make_pair(0.0f, 1.0f), std::make_pair(0.1f, 1.0f), std::make_pair(0.3f, 1.0f),
                                            std::make_pair(0.5f, 1.0f), std::make_pair(0.45f, 1.0f), std::make_pair(0.35f, 1.0f), std::make_pair(0.0f, 1.0f) };   
    int sample_option;
    std::pair<float, float> iou;
    float min_iou, max_iou;
    in_width = _meta_crop_param->in_width;
    in_height = _meta_crop_param->in_height;
    bool invalid_bboxes = true;
    _enitire_iou = true;
    BoundingBoxCord crop_box, jth_box;
    jth_box.x = jth_box.y = jth_box.w = jth_box.h = 0; // Initializing to supress warnings. 
    for (int i = 0; i < _batch_size; i++)
    {
        int bb_count = input_meta_data->get_bb_labels_batch()[i].size();
        std::vector<int> labels_buf(bb_count);
        memcpy(labels_buf.data(), input_meta_data->get_bb_labels_batch()[i].data(), sizeof(int) * bb_count);
        std::vector<float> coords_buf(bb_count * 4);
        memcpy(coords_buf.data(), input_meta_data->get_bb_cords_batch()[i].data(), input_meta_data->get_bb_cords_batch()[i].size() * sizeof(BoundingBoxCord));
        crop_box.h = _crop_height_val[i];
        crop_box.w = _crop_width_val[i];
        crop_box.x = _x1_val[i];
        crop_box.y = _y1_val[i];
        while (true)
        {
            sample_option = dis(gen);
            iou =  IOU[sample_option];
            min_iou = iou.first;
            max_iou = iou.second;
            if (!sample_option)
            {
                crop_box.x = 0;
                crop_box.y = 0;
                crop_box.h = in_height[i];
                crop_box.w = in_width[i];
                break;
            }

            for (int j = 0; j < _num_of_attempts; j++)
            {
                x_drift_factor->renew();
                float factor = 0.3f;
                auto w_factor = factor + (x_drift_factor->get() * (1 - factor));
                crop_box.w = w_factor * in_width[i];
                y_drift_factor->renew();
                y_drift_factor->renew();
                auto h_factor = factor + (y_drift_factor->get() * (1 - factor));
                crop_box.h = h_factor * in_height[i];
                //aspect ratio check
                if ((crop_box.w / crop_box.h < 0.5) || (crop_box.w / crop_box.h > 2.))
                    continue;
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
                float bb_iou = BBoxIntersectionOverUnion(jth_box, crop_box, _enitire_iou); 
                if (bb_iou < min_iou || bb_iou > max_iou )
                {
                    invalid_bboxes = true;
                    break;
                }
            }  

            if (invalid_bboxes)
                continue;
            int valid_bbox_count = 0;
            auto left = crop_box.x, top = crop_box.y, right = crop_box.x + crop_box.w, bottom = crop_box.y + crop_box.h;
            for (int j = 0; j < bb_count; j++)
            {
                int m = j * 4;
                auto x_c = 0.5f * (2 * coords_buf[m] + coords_buf[m + 2]);
                auto y_c = 0.5f * (2 * coords_buf[m + 1] + coords_buf[m + 3]);
                if ((x_c >= left) && (x_c <= right) && (y_c >= top) && (y_c <= bottom))
                    valid_bbox_count++;
            }
            if (valid_bbox_count == 0)
                continue;
            break;
        } // while loop
        BoundingBoxCords bb_coords;
        BoundingBoxLabels bb_labels;
        for (int j = 0; j < bb_count; j++)
        {
            int m = j*4;
            BoundingBoxCord box;
            box.x = coords_buf[m++];
            box.y = coords_buf[m++];
            box.w = coords_buf[m++];
            box.h = coords_buf[m++];
            if(sample_option)
            {
                auto x_c = 0.5f * (box.x + box.x + box.w);
                auto y_c = 0.5f * (box.y + box.y + box.h);
                if ((x_c >= crop_box.x) && (x_c <= crop_box.x + crop_box.w) && (y_c >= crop_box.y) && (y_c <= crop_box.y + crop_box.h))
                {
                    float bb_iou = BBoxIntersectionOverUnion(jth_box, crop_box, _enitire_iou); 
                    if (bb_iou >= min_iou && bb_iou <= max_iou)
                    {
                        float xA = std::max(crop_box.x, box.x);
                        float yA = std::max(crop_box.y, box.y);
                        float xB = std::min(crop_box.x + crop_box.w, box.x + box.w);
                        float yB = std::min(crop_box.y + crop_box.h, box.y + box.h);
                        box.x =  xA - crop_box.x;
                        box.y =  yA - crop_box.y;
                        box.w = xB - xA;
                        box.h = yB - yA;
                        bb_coords.push_back(box);
                        bb_labels.push_back(labels_buf[j]);
                    }
                }    
            }
            else
            {
                bb_coords.push_back(box);
                bb_labels.push_back(labels_buf[j]);
            }
        }
        #if 0
        std::cout << " BBox Information!!!!!! " << std::endl;
        std::cout << " number of input bboxes " << bb_count << std::endl;
        for (int j = 0; j < bb_count; j++)
        {
            int m = j*4;
            BoundingBoxCord box;
            box.x = coords_buf[m++];
            box.y = coords_buf[m++];
            box.w = coords_buf[m++];
            box.h = coords_buf[m++];
            std::cout << "Input Box  " << box.x << " " << box.y << " " << box.w << " " << box.h << " Label :" << labels_buf[j] << std::endl;  
        }
        std::cout << " number of output bboxes " << bb_coords.size() << std::endl;
        for(int j = 0 ; j < bb_coords.size(); j++)
        {
            BoundingBoxCord box = bb_coords[j];
            std::cout << "Output Box  " << box.x << " " << box.y << " " << box.w << " " << box.h << " Label :" << bb_labels[j] << std::endl;  
        }
        std::cout << " BBox Information!!!!!! " << std::endl;
        #endif
        
        _x1_val[i] = crop_box.x;
        _y1_val[i] = crop_box.y;
        _crop_width_val[i] = crop_box.w;
        _crop_height_val[i] = crop_box.h;
        _x2_val[i] = _x1_val[i] + crop_box.w;
        _y2_val[i] = _y2_val[i] + crop_box.h;
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
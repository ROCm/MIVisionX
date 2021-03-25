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

inline double BBoxIntersectionOverUnion(const BoundingBoxCord &box1, const BoundingBoxCord &box2, bool is_iou = false)
{
    double iou;
    float xA = std::max(box1.x, box2.x);
    float yA = std::max(box1.y, box2.y);
    float xB = std::min(box1.x + box1.w, box2.x + box2.w);
    float yB = std::min(box1.y + box1.h, box2.y + box2.h);

    float intersection_area = std::max((float)0.0, xB - xA) * std::max((float)0.0, yB - yA);

    float box1_area = box1.h * box1.w;
    float box2_area = box2.h * box2.w;

    if (is_iou)
        iou = intersection_area / float(box1_area + box2_area - intersection_area);
    else
        iou = intersection_area / float(box1_area);

    return iou;
}

void BoundingBoxGraph::update_random_bbox_meta_data(CropCordBatch *_random_bbox_crop_cords_data, MetaDataBatch *input_meta_data, decoded_image_info decode_image_info)
{
    std::vector<uint32_t> original_height = decode_image_info._original_height;
    std::vector<uint32_t> original_width = decode_image_info._original_width;
    std::vector<uint32_t> roi_width = decode_image_info._roi_width;
    std::vector<uint32_t> roi_height = decode_image_info._roi_height;
    auto crop_cords = _random_bbox_crop_cords_data->get_bb_cords_batch();
    for (int i = 0; i < input_meta_data->size(); i++)
    {
        auto bb_count = input_meta_data->get_bb_labels_batch()[i].size();
        int labels_buf[bb_count];
        float coords_buf[bb_count * 4];
        memcpy(labels_buf, input_meta_data->get_bb_labels_batch()[i].data(), sizeof(int) * bb_count);
        memcpy(coords_buf, input_meta_data->get_bb_cords_batch()[i].data(), input_meta_data->get_bb_cords_batch()[i].size() * sizeof(BoundingBoxCord));
        BoundingBoxCords bb_coords;
        BoundingBoxLabels bb_labels;
        BoundingBoxCord crop_box;
        crop_box.x = crop_cords[i]->crop_x;
        crop_box.y = crop_cords[i]->crop_y;
        crop_box.w = crop_cords[i]->crop_width;
        crop_box.h = crop_cords[i]->crop_height;
        for (uint j = 0; j < bb_count; j++)
        {
            int m = j * 4; // change if required

            //Mask Criteria
            auto left = int(crop_box.x);
            auto top = crop_box.y;
            auto right = int(crop_box.x) + int(crop_box.w);
            auto bottom = crop_box.y + crop_box.h;

            BoundingBoxCord box;
            box.x = coords_buf[m];
            box.y = coords_buf[m + 1];
            box.w = coords_buf[m + 2];
            box.h = coords_buf[m + 3];
            auto x_c = 0.5f * (2 * box.x + box.w);
            auto y_c = 0.5f * (2 * box.y + box.h);
            if ((x_c > left) && (x_c < right) && (y_c > top) && (y_c < bottom))
            {
                float xA = std::max(crop_box.x, box.x);
                float yA = std::max(crop_box.y, box.y);
                float xB = std::min(crop_box.x + crop_box.w, box.x + box.w);
                float yB = std::min(crop_box.y + crop_box.h, box.y + box.h);
                box.x = xA - crop_box.x;
                box.y = yA - crop_box.y;
                box.w = xB - xA;
                box.h = yB - yA;

                bb_coords.push_back(box);
                bb_labels.push_back(labels_buf[j]);                
            }
        }
        if (bb_coords.size() == 0)
        {
            std::cerr << "Bounding box co-ordinates not found in the image";
        }
        input_meta_data->get_bb_cords_batch()[i] = bb_coords;
        input_meta_data->get_bb_labels_batch()[i] = bb_labels;
    }
}

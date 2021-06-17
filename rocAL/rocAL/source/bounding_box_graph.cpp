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

//not required since the bbox are normalized in the very beggining -> remove the call in master graph also
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
        temp_box.l = temp_box.t = temp_box.r = temp_box.b = 0;
        BoundingBoxLabels bb_labels;
        int m = 0;
        for (uint j = 0; j < bb_count; j++)
        {
            BoundingBoxCord box;
            float temp_l, temp_t;
            temp_l = (coords_buf[m++] * _dst_to_src_width_ratio);
            temp_t = (coords_buf[m++] * _dst_to_src_height_ratio);
            box.l = std::max(temp_l,0.0f);
            box.t = std::max(temp_t,0.0f);
            box.r = (coords_buf[m++] * _dst_to_src_width_ratio);
            box.b = (coords_buf[m++] * _dst_to_src_height_ratio);
            bb_coords.push_back(box);
        }
        if (bb_coords.size() == 0)
        {
            bb_coords.push_back(temp_box);
            bb_labels.push_back(0);
        }
        input_meta_data->get_bb_cords_batch()[i] = bb_coords;
    }
}

inline double ssd_BBoxIntersectionOverUnion(const BoundingBoxCord &box1, const BoundingBoxCord &box2, bool is_iou = false)
{
    double iou;
    float xA = std::max(box1.l, box2.l);
    float yA = std::max(box1.t, box2.t);
    float xB = std::min(box1.r, box2.r);
    float yB = std::min(box1.b, box2.b);

    float intersection_area = std::max((float)0.0, xB - xA) * std::max((float)0.0, yB - yA);
    float box1_area = (box1.b - box1.t) * (box1.r - box1.l);
    float box2_area = (box2.b - box2.t) * (box2.r - box2.l);

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
        crop_box.l = crop_cords[i]->crop_left;
        crop_box.t = crop_cords[i]->crop_top;
        crop_box.r = crop_cords[i]->crop_right;
        crop_box.b = crop_cords[i]->crop_bottom;
        // std::cout<<"Original <Widthx,Height>"<<original_width[i]<<" X "<<original_height[i];
        // std::cout  << " In bounding box graph ::crop<l,t,r,b>: " << crop_box.l << " X " << crop_box.t << " X " << crop_box.r << " X " << crop_box.b << std::endl;

        for (uint j = 0; j < bb_count; j++)
        {
            int m = j * 4; // change if required
            //Mask Criteria
            BoundingBoxCord box;
            box.l = coords_buf[m];
            box.t = coords_buf[m + 1];
            box.r = coords_buf[m + 2];
            box.b = coords_buf[m + 3];
            // std::cout  << " In bounding box graph ::valid BBOXES ::bboxes<l,t,r,b>: " << box.l << " X " << box.t << " X " << box.r << " X " << box.b << std::endl;

            auto x_c = 0.5 * (box.l + box.r);
            auto y_c = 0.5 * (box.t + box.b);
            if ((x_c > crop_box.l) && (x_c < crop_box.r) && (y_c > crop_box.t) && (y_c < crop_box.b))
            {
                float xA = std::max(crop_box.l, box.l);
                float yA = std::max(crop_box.t, box.t);
                float xB = std::min(crop_box.r, box.r);
                float yB = std::min(crop_box.b, box.b);
                box.l = (xA - crop_box.l) / (crop_box.r - crop_box.l);
                box.t = (yA - crop_box.t) / (crop_box.b - crop_box.t);
                box.r = (xB - crop_box.l) / (crop_box.r - crop_box.l);
                box.b = (yB - crop_box.t) / (crop_box.b - crop_box.t);
                // std::cout<<"\nn bounding box grpah: Box Co-ordinates lxtxrxb::\t"<<box.l<<"x\t"<<box.t<<"x\t"<<box.r<<"x\t"<<box.b<<"x\t"<<std::endl;

                bb_coords.push_back(box);
                bb_labels.push_back(labels_buf[j]);
            }
        }
        if (bb_coords.size() == 0)
        {
            std::cerr << "Bounding box co-ordinates not found in the image";
            exit(0);
        }
        input_meta_data->get_bb_cords_batch()[i] = bb_coords;
        input_meta_data->get_bb_labels_batch()[i] = bb_labels;
    }
}

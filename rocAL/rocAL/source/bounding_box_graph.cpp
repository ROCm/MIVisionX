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
    // std::cout << "\tl:" << xA << "\t t:" << yA << "\tr:" << xB << "\t b:" << yB;
    float intersection_area = std::max((float)0.0, xB - xA) * std::max((float)0.0, yB - yA);
    // std::cout << "\n intersection area: " << intersection_area;
    float box1_area = (box1.b - box1.t) * (box1.r - box1.l);
    float box2_area = (box2.b - box2.t) * (box2.r - box2.l);
    // std::cout << "\n box1_area:" << box1_area;
    // std::cout << "\n box2_area:" << box2_area;

    if (is_iou)
    {
        iou = intersection_area / float(box1_area + box2_area - intersection_area);
        // std::cout << "\niou inside iou:" << iou;
    }
    else
        iou = intersection_area / float(box1_area);

    return iou;
}

void BoundingBoxGraph::update_random_bbox_meta_data(MetaDataBatch *input_meta_data, decoded_image_info decode_image_info, crop_image_info crop_image_info)
{
    std::vector<uint32_t> original_height = decode_image_info._original_height;
    std::vector<uint32_t> original_width = decode_image_info._original_width;
    std::vector<uint32_t> roi_width = decode_image_info._roi_width;
    std::vector<uint32_t> roi_height = decode_image_info._roi_height;
    auto crop_cords = crop_image_info._crop_image_coords;
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
        crop_box.l = crop_cords[i][0];
        crop_box.t = crop_cords[i][1];
        crop_box.r = crop_box.l + crop_cords[i][2];
        crop_box.b = crop_box.t + crop_cords[i][3];
        // std::cout << "\n BB count" << bb_count;
        for (uint j = 0; j < bb_count; j++)
        {
            int m = j * 4; // change if required
            //Mask Criteria
            BoundingBoxCord box;
            box.l = coords_buf[m];
            box.t = coords_buf[m + 1];
            box.r = coords_buf[m + 2];
            box.b = coords_buf[m + 3];
            // std::cout << "\nbox_l:" << box.l << "\tbox_t:" << box.t << "\tbox_r:" << box.r << "\tbox_b:" << box.b;
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

inline void calculate_ious_for_box(float *ious, BoundingBoxCord box, std::vector<float> anchors)
{
    BoundingBoxCord _anchor;
    _anchor.l = anchors[0];
    _anchor.t = anchors[1];
    _anchor.r = anchors[2];
    _anchor.b = anchors[3];
    ious[0] = ssd_BBoxIntersectionOverUnion(box, _anchor,true);
    int best_idx = 0;
    float best_iou = ious[0];

    for (unsigned int anchor_idx = 1; anchor_idx < (anchors.size() / 4); anchor_idx++)
    {
        int m = anchor_idx * 4;
        BoundingBoxCord _anchor;
        _anchor.l = anchors[m];
        _anchor.t = anchors[m + 1];
        _anchor.r = anchors[m + 2];
        _anchor.b = anchors[m + 3];
        float x = ssd_BBoxIntersectionOverUnion(box, _anchor, true);
        ious[anchor_idx] = x;
        if (ious[anchor_idx] > best_iou)
        {
            best_iou = ious[anchor_idx];
            best_idx = anchor_idx;
        }
    }
    // For best default box matched with current object let iou = 2, to make sure there is a match,
    // as this object will be the best (highest IoU), for this default box
    ious[best_idx] = 2.;
}

inline int find_best_box_for_anchor(unsigned anchor_idx, const std::vector<float> &ious, unsigned num_boxes, unsigned anchors_size)
{
    unsigned best_idx = 0;
    float best_iou = ious[anchor_idx];
    for (unsigned bbox_idx = 1; bbox_idx < num_boxes; ++bbox_idx)
    {
        if (ious[bbox_idx * anchors_size + anchor_idx] >= best_iou)
        {
            best_iou = ious[bbox_idx * anchors_size + anchor_idx];
            best_idx = bbox_idx;
        }
    }
    return best_idx;
}

void BoundingBoxGraph::update_box_encoder_meta_data(std::vector<float> anchors, pMetaDataBatch full_batch_meta_data, float criteria, bool offset, float scale, std::vector<float> means, std::vector<float> stds)
{
    #pragma omp parallel for 
    for (int i = 0; i < full_batch_meta_data->size(); i++)
    {
        auto bb_count = full_batch_meta_data->get_bb_labels_batch()[i].size();
        int labels_buf[bb_count];
        float coords_buf[bb_count * 4];
        memcpy(labels_buf, full_batch_meta_data->get_bb_labels_batch()[i].data(), sizeof(int) * bb_count);
        memcpy(coords_buf, full_batch_meta_data->get_bb_cords_batch()[i].data(), full_batch_meta_data->get_bb_cords_batch()[i].size() * sizeof(BoundingBoxCord));
        BoundingBoxCords encoded_bb;
        BoundingBoxLabels encoded_labels;
        BoundingBoxCords bb_coords;
        BoundingBoxLabels bb_labels;
        unsigned anchors_size = anchors.size() / 4; // divide the anchors_size by 4 to get the total number of anchors
        //Calculate Ious
        //ious size - bboxes count x anchors count
        std::vector<float> ious(bb_count * anchors_size);
        bb_coords.resize(bb_count);
        bb_labels.resize(bb_count);
        encoded_bb.resize(anchors_size);
        encoded_labels.resize(anchors_size);

        for (uint bb_idx = 0; bb_idx < bb_count; bb_idx++)
        {
            int m = bb_idx * 4;
            BoundingBoxCord box;
            box.l = coords_buf[m];
            box.t = coords_buf[m + 1];
            box.r = coords_buf[m + 2];
            box.b = coords_buf[m + 3];
            auto iou_rows = ious.data() + (bb_idx * (anchors_size));
            calculate_ious_for_box(iou_rows, box, anchors);
            bb_coords[bb_idx] = box;
            bb_labels[bb_idx] = labels_buf[bb_idx];
        }
        
        // Depending on the matches ->place the best bbox instead of the corresponding anchor_idx in anchor
        for (unsigned anchor_idx = 0; anchor_idx < anchors_size; anchor_idx++)
        {
            int m = anchor_idx * 4;
            BoundingBoxCord box_bestidx, _anchor, _anchor_xcyxwh;
            _anchor.l = anchors[m];
            _anchor.t = anchors[m + 1];
            _anchor.r = anchors[m + 2];
            _anchor.b = anchors[m + 3];
            const auto best_idx = find_best_box_for_anchor(anchor_idx, ious, bb_count, anchors_size);
            // Filter matches by criteria
            if (ious[(best_idx * anchors_size) + anchor_idx] > criteria) //Its a match
            {
                //YTD: Need to add a new structure for xc,yc,w,h similar to l,t,r,b as a part of metadata
                    box_bestidx.l = 0.5 * (bb_coords.at(best_idx).l + bb_coords.at(best_idx).r); //xc
                    box_bestidx.t = 0.5 * (bb_coords.at(best_idx).t + bb_coords.at(best_idx).b); //yc
                    box_bestidx.r = (-bb_coords.at(best_idx).l + bb_coords.at(best_idx).r);      //w
                    box_bestidx.b = (-bb_coords.at(best_idx).t + bb_coords.at(best_idx).b);      //h

                if (offset)
                {
                    box_bestidx.l *= scale; //xc
                    box_bestidx.t *= scale; //yc
                    box_bestidx.r *= scale; //w
                    box_bestidx.b *= scale; //h
                    //YTD: Need to add a new structure for xc,yc,w,h similar to l,t,r,b as a part of metadata
                    _anchor_xcyxwh.l = 0.5 * (_anchor.l + _anchor.r) * scale; //xc
                    _anchor_xcyxwh.t = 0.5 * (_anchor.t + _anchor.b) * scale; //yc
                    _anchor_xcyxwh.r = (-_anchor.l + _anchor.r) * scale;      //w
                    _anchor_xcyxwh.b = (-_anchor.t + _anchor.b) * scale;      //h

                    box_bestidx.l = ((box_bestidx.l - _anchor_xcyxwh.l) / _anchor_xcyxwh.r - means[0]) / stds[0];
                    box_bestidx.t = ((box_bestidx.t - _anchor_xcyxwh.t) / _anchor_xcyxwh.b - means[1]) / stds[1];
                    box_bestidx.r = (std::log(box_bestidx.r / _anchor_xcyxwh.r) - means[2]) / stds[2];
                    box_bestidx.b = (std::log(box_bestidx.b / _anchor_xcyxwh.b) - means[3]) / stds[3];
                    encoded_bb[anchor_idx] = box_bestidx;
                    encoded_labels[anchor_idx] = bb_labels.at(best_idx);
                }
                else
                {
                    encoded_bb[anchor_idx] = box_bestidx;
                    encoded_labels[anchor_idx] = bb_labels.at(best_idx);
                }
            }
            else // Not a match
            {
                if (offset)
                {
                    _anchor_xcyxwh.l =_anchor_xcyxwh.t = _anchor_xcyxwh.r = _anchor_xcyxwh.b = 0;
                    encoded_bb[anchor_idx] = _anchor_xcyxwh;
                    encoded_labels[anchor_idx] = 0;
                }
                else
                {
                    //YTD: Need to add a new structure for xc,yc,w,h similar to l,t,r,b as a part of metadata
                    _anchor_xcyxwh.l = 0.5 * (_anchor.l + _anchor.r); //xc
                    _anchor_xcyxwh.t = 0.5 * (_anchor.t + _anchor.b); //yc
                    _anchor_xcyxwh.r = (-_anchor.l + _anchor.r);      //w
                    _anchor_xcyxwh.b = (-_anchor.t + _anchor.b);      //h
                    encoded_bb[anchor_idx] = _anchor_xcyxwh;
                    encoded_labels[anchor_idx] = 0;
                }
            }
        }
        bb_coords.clear();
        bb_labels.clear();
        full_batch_meta_data->get_bb_cords_batch()[i] = encoded_bb;
        full_batch_meta_data->get_bb_labels_batch()[i] = encoded_labels;
        encoded_bb.clear();
        encoded_labels.clear();
    }
}

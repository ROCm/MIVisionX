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
    {
        iou = intersection_area / float(box1_area + box2_area - intersection_area);
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
        for (uint j = 0; j < bb_count; j++)
        {
            int m = j * 4; // change if required
            //Mask Criteria
            BoundingBoxCord box;
            box.l = coords_buf[m];
            box.t = coords_buf[m + 1];
            box.r = coords_buf[m + 2];
            box.b = coords_buf[m + 3];
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

inline void calculate_ious_for_box(float *ious, BoundingBoxCord box, BoundingBoxCords* anchors)
{

    BoundingBoxCord anchor;
    anchor = (*anchors)[0];
    ious[0] = ssd_BBoxIntersectionOverUnion(box, anchor,true);
    int best_idx = 0;
    float best_iou = ious[0];

    for (unsigned int anchor_idx = 1; anchor_idx < ((*anchors).size()); anchor_idx++)
    {
        anchor = (*anchors)[anchor_idx];
        ious[anchor_idx] = ssd_BBoxIntersectionOverUnion(box, anchor, true);
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
        BoundingBoxCords_xcycwh encoded_bb;
        BoundingBoxLabels encoded_labels;
        BoundingBoxCords bb_coords;
        BoundingBoxLabels bb_labels;
        BoundingBoxCords* anchors_cast = (BoundingBoxCords *)&anchors;
        unsigned anchors_size = (*anchors_cast).size(); 
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
            calculate_ious_for_box(iou_rows, box, anchors_cast);
            bb_coords[bb_idx] = box;
            bb_labels[bb_idx] = labels_buf[bb_idx];
        }
        
        // Depending on the matches ->place the best bbox instead of the corresponding anchor_idx in anchor
        for (unsigned anchor_idx = 0; anchor_idx < anchors_size; anchor_idx++)
        {
            BoundingBoxCord anchor;
            BoundingBoxCord_xcycwh _anchor_xcycwh, box_bestidx;
            anchor = (*anchors_cast)[anchor_idx];
            const auto best_idx = find_best_box_for_anchor(anchor_idx, ious, bb_count, anchors_size);
            // Filter matches by criteria
            if (ious[(best_idx * anchors_size) + anchor_idx] > criteria) //Its a match
            {
                    //Convert the "ltrb" format to "xcycwh"
                    box_bestidx.xc = 0.5 * (bb_coords.at(best_idx).l + bb_coords.at(best_idx).r); 
                    box_bestidx.yc = 0.5 * (bb_coords.at(best_idx).t + bb_coords.at(best_idx).b); 
                    box_bestidx.w = (-bb_coords.at(best_idx).l + bb_coords.at(best_idx).r);      
                    box_bestidx.h = (-bb_coords.at(best_idx).t + bb_coords.at(best_idx).b);      

                if (offset)
                {
                    box_bestidx.xc *= scale; 
                    box_bestidx.yc *= scale; 
                    box_bestidx.w *= scale; 
                    box_bestidx.h *= scale; 
                    
                    _anchor_xcycwh.xc = 0.5 * (anchor.l + anchor.r) * scale; 
                    _anchor_xcycwh.yc = 0.5 * (anchor.t + anchor.b) * scale; 
                    _anchor_xcycwh.w = (-anchor.l + anchor.r) * scale;      
                    _anchor_xcycwh.h = (-anchor.t + anchor.b) * scale;      

                    // Reference for offset calculation using GT boxes & anchor boxes in <xc,yc,w,h> format
                    // https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#predictions-vis-%C3%A0-vis-priors
                    box_bestidx.xc = ((box_bestidx.xc- _anchor_xcycwh.xc) / _anchor_xcycwh.w - means[0]) / stds[0];
                    box_bestidx.yc = ((box_bestidx.yc - _anchor_xcycwh.yc) / _anchor_xcycwh.h - means[1]) / stds[1];
                    box_bestidx.w = (std::log(box_bestidx.w / _anchor_xcycwh.w) - means[2]) / stds[2];
                    box_bestidx.h = (std::log(box_bestidx.h / _anchor_xcycwh.h) - means[3]) / stds[3];
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
                    _anchor_xcycwh.xc =_anchor_xcycwh.yc = _anchor_xcycwh.w = _anchor_xcycwh.h = 0;
                    encoded_bb[anchor_idx] = _anchor_xcycwh;
                    encoded_labels[anchor_idx] = 0;
                }
                else
                {
                    //Convert the "ltrb" format to "xcycwh"
                    _anchor_xcycwh.xc = 0.5 * (anchor.l + anchor.r); //xc
                    _anchor_xcycwh.yc = 0.5 * (anchor.t + anchor.b); //yc
                    _anchor_xcycwh.w = (-anchor.l + anchor.r);      //w
                    _anchor_xcycwh.h = (-anchor.t + anchor.b);      //h
                    encoded_bb[anchor_idx] = _anchor_xcycwh;
                    encoded_labels[anchor_idx] = 0;
                }
            }
        }
        bb_coords.clear();
        bb_labels.clear();
        BoundingBoxCords * encoded_bb_ltrb = (BoundingBoxCords*)&encoded_bb;
        full_batch_meta_data->get_bb_cords_batch()[i] = (*encoded_bb_ltrb);
        full_batch_meta_data->get_bb_labels_batch()[i] = encoded_labels;
        encoded_bb.clear();
        encoded_labels.clear();
    }
}

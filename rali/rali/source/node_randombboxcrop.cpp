#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_randombboxcrop.h"
#include "exception.h"

RandomBBoxCropNode::RandomBBoxCropNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) : Node(inputs, outputs),
                                                                                                                  _dest_width(_outputs[0]->info().width()),
                                                                                                                  _dest_height(_outputs[0]->info().height_batch())
{
    _crop_param = std::make_shared<RaliRandomCropParam>(_batch_size);
    _is_ssd = true;
}

void RandomBBoxCropNode::create_node()
{
    _crop_width_val.resize(_batch_size);
    _crop_height_val.resize(_batch_size);
    _x1_val.resize(_batch_size);
    _y1_val.resize(_batch_size);
    _x2_val.resize(_batch_size);
    _y2_val.resize(_batch_size);
    _iou_range.resize(_batch_size);
    if (_node)
        return;

    if (_dest_width == 0 || _dest_height == 0)
        THROW("Uninitialized destination dimension")

    _crop_param->create_array(_graph);
    _node = vxExtrppNode_CropPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _crop_param->cropw_arr,
                                _crop_param->croph_arr, _crop_param->x1_arr, _crop_param->y1_arr, _batch_size);

    vx_status status;
    if ((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Error adding the crop resize node (vxExtrppNode_ResizeCropbatchPD    ) failed: " + TOSTR(status))
}

inline double ssd_BBoxIntersectionOverUnion(const BoundingBoxCord &box1, const BoundingBoxCord &box2, bool is_iou = false)
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

void RandomBBoxCropNode::update_node()
{
    _crop_param->set_image_dimensions(_inputs[0]->info().get_roi_width_vec(), _inputs[0]->info().get_roi_height_vec());
    _crop_param->update_array();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 6);

    const std::vector<std::pair<bool, float>> sample_options = {std::make_pair(true, 0.0f), std::make_pair(false, 0.1f), std::make_pair(false, 0.15f),
                                                                std::make_pair(false, 0.5f), std::make_pair(false, 0.6f), std::make_pair(false, 0.75f), std::make_pair(false, 0.9f)};
    int sample_option;
    std::pair<bool, float> option;
    float min_iou, max_iou;
    in_width = _crop_param->in_width;
    in_height = _crop_param->in_height;
    bool invalid_bboxes = true;
    _entire_iou = true;
    bool _overlap_iou = false;
    BoundingBoxCord crop_box, jth_box;
    int bb_count;
    Parameter<float> *x_drift_factor = _crop_param->get_x_drift_factor();
    Parameter<float> *y_drift_factor = _crop_param->get_y_drift_factor();
    Parameter<float> *area_factor = _crop_param->get_area_factor();
    Parameter<float> *aspect_ratio_factor = _crop_param->get_aspect_ratio();
    _x1_val = _crop_param->get_x1_arr_val();
    _y1_val = _crop_param->get_y1_arr_val();
    _crop_width_val = _crop_param->get_cropw_arr_val();
    _crop_height_val = _crop_param->get_croph_arr_val();
    for (uint i = 0; i < _batch_size; i++)
    {
        // std::vector<std::vector<int>> out_labels;
        // std::vector<>
        bool crop_success = false;
        int count = 0;
        bb_count = _meta_data_info->get_bb_labels_batch()[i].size();
        std::vector<int> labels_buf(bb_count);
        memcpy(labels_buf.data(), _meta_data_info->get_bb_labels_batch()[i].data(), sizeof(int) * bb_count);
        std::vector<float> coords_buf(bb_count * 4);
        memcpy(coords_buf.data(), _meta_data_info->get_bb_cords_batch()[i].data(), _meta_data_info->get_bb_cords_batch()[i].size() * sizeof(BoundingBoxCord));
        crop_box.h = _crop_height_val[i];
        crop_box.w = _crop_width_val[i];
        crop_box.x = _x1_val[i];
        crop_box.y = _y1_val[i];
        // Got BBOX Information of the image, try to get a crop
        while (!crop_success && (_num_of_attempts < 0 || count < _num_of_attempts))
        {
            sample_option = dis(gen);
            option = sample_options[sample_option];
            _iou_range[i] = option;
            _no_crop = option.first;
            min_iou = option.second;
            //sample_option = 0;
            if (_no_crop)
            {
                crop_box.x = 0;
                crop_box.y = 0;
                crop_box.h = in_height[i];
                crop_box.w = in_width[i];
                break;
            }

            if (_has_shape)
            {
                crop_box.w = _crop_width;  // Given By user
                crop_box.h = _crop_height; // Given By user
            }
            else // If it has no shape, then area and aspect ratio thing should be provided
            {
                /*for (int j = 0; j < _num_of_attempts; j++)
                {
                    //_scale_factor->renew(); //_scale_factor is a random_number picked from [min-max] range
                    //auto w_factor = _scale_factor->get();
                    
                    crop_box.w = w_factor * in_width[i];
                    //_scale_factor->renew(); //_scale_factor is a random_number picked from [min-max] range
                    //auto h_factor = _scale_factor->get();
                    crop_box.h = h_factor * in_height[i];
                    if ((crop_box.w / crop_box.h < 0.5) || (crop_box.w / crop_box.h > 2.))
                        continue;
                    break;
                }
                if ((crop_box.w / crop_box.h < 0.5) || (crop_box.w / crop_box.h > 2.))
                    continue;*/
                for (int j = 0; j < _num_of_attempts; j++)
                {
                    area_factor->renew(); //_scale_factor is a random_number picked from [min-max] range 
                    auto w_factor =  area_factor->get();
                    crop_box.w = w_factor * in_width[i];
                    area_factor->renew(); //_scale_factor is a random_number picked from [min-max] range 
                    auto h_factor = area_factor->get();
                    crop_box.h = h_factor * in_height[i];
                    //aspect ratio check
                    aspect_ratio_factor->renew(); // _aspect ratio factor is a random number piced in [minx/y and maxx/y] params
                    auto min_ar = aspect_ratio_factor->get();
                    aspect_ratio_factor->renew();
                    auto max_ar = aspect_ratio_factor->get();
                    if(min_ar == max_ar) {min_ar = 0.5; max_ar = 2.0;} // No need to try more to get this..in stead go ached with the input min and max
                    if(min_ar > max_ar) std::swap(min_ar, max_ar);
                    if ((crop_box.w / crop_box.h < min_ar) || (crop_box.w / crop_box.h > max_ar))
                        continue;
                    break;
                }
                // if ((crop_box.w / crop_box.h < 0.5) || (crop_box.w / crop_box.h > 2.)) if it is not stilll successfull aszpect ratio - force somethng default
                //     continue;
                //Got the crop;
            }

            x_drift_factor->renew();
            x_drift_factor->renew();
            y_drift_factor->renew(); // Get the random parameters between 0 - 1 (Uniform random)
            crop_box.x = static_cast<size_t>(x_drift_factor->get() * (in_width[i] - crop_box.w));
            crop_box.y = static_cast<size_t>(y_drift_factor->get() * (in_height[i] - crop_box.h));
            bool entire_iou = !_overlap_iou; //
            if (_all_boxes_overlap)
            {
                for (int j = 0; j < bb_count; j++)
                {
                    int m = j * 4;
                    jth_box.x = coords_buf[m];
                    jth_box.y = coords_buf[m + 1];
                    jth_box.w = coords_buf[m + 2];
                    jth_box.h = coords_buf[m + 3];
                    float bb_iou = ssd_BBoxIntersectionOverUnion(jth_box, crop_box, entire_iou);
                    if (bb_iou < min_iou)
                    {
                        invalid_bboxes = true;
                        break;
                    }
                }
                if (invalid_bboxes)
                    continue;
            }
            else // at lease one box shoud overlap
            {

                for (int j = 0; j < bb_count; j++)
                {
                    int m = j * 4;
                    jth_box.x = coords_buf[m];
                    jth_box.y = coords_buf[m + 1];
                    jth_box.w = coords_buf[m + 2];
                    jth_box.h = coords_buf[m + 3];
                    float bb_iou = ssd_BBoxIntersectionOverUnion(jth_box, crop_box, entire_iou);
                    if (bb_iou > min_iou)
                    {
                        invalid_bboxes = true;
                        break;
                    }
                }
                if (invalid_bboxes)
                    continue;
            }

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
            crop_success = true;
        }                                     // while loop
        _x1_val[i] = crop_box.x;              // L
        _y1_val[i] = crop_box.y;              // T
        _crop_width_val[i] = crop_box.w;      // W
        _crop_height_val[i] = crop_box.h;     // H
        _x2_val[i] = _x1_val[i] + crop_box.w; // R
        _y2_val[i] = _y2_val[i] + crop_box.h; // B
    }
    vxCopyArrayRange((vx_array)_crop_param->cropw_arr, 0, _batch_size, sizeof(uint), _crop_width_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_crop_param->croph_arr, 0, _batch_size, sizeof(uint), _crop_height_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_crop_param->x1_arr, 0, _batch_size, sizeof(uint), _x1_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_crop_param->y1_arr, 0, _batch_size, sizeof(uint), _y1_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    //_outputs[0]->update_image_roi(_crop_width_val, _crop_height_val);
}

void RandomBBoxCropNode::init(FloatParam *crop_area_factor, FloatParam *crop_aspect_ratio, FloatParam *x_drift, FloatParam *y_drift, int num_of_attempts, int all_boxes_overlap, int no_crop, int has_shape, int crop_width, int crop_height)
{
    _crop_param->set_x_drift_factor(core(x_drift));
    _crop_param->set_y_drift_factor(core(y_drift));
    _crop_param->set_area_factor(core(crop_area_factor));
    _crop_param->set_aspect_ratio(core(crop_aspect_ratio));
    _num_of_attempts = num_of_attempts;
    _all_boxes_overlap = all_boxes_overlap;
    _no_crop = no_crop;
    _has_shape = has_shape;
    _crop_width = crop_width;
    _crop_height = crop_height;
}

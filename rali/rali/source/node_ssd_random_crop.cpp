#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_ssd_random_crop.h"
#include "exception.h"

SSDRandomCropNode::SSDRandomCropNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) : Node(inputs, outputs),
                                                                                                          _dest_width(_outputs[0]->info().width()),
                                                                                                          _dest_height(_outputs[0]->info().height_batch())
{
    _crop_param = std::make_shared<RaliRandomCropParam>(_batch_size);
    _is_ssd     = true;
}

void SSDRandomCropNode::create_node()
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

    if(is_iou)
        iou = intersection_area / float(box1_area + box2_area - intersection_area);
    else
        iou = intersection_area / float(box1_area);

    return iou;
}

void SSDRandomCropNode::update_node()
{
    _crop_param->set_image_dimensions(_inputs[0]->info().get_roi_width_vec(), _inputs[0]->info().get_roi_height_vec());
    _crop_param->update_array();
    // std::cerr<<"\n batch_size:: "<<_batch_size<<"\n meta array size:: "<<_meta_data_info->size();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 6);
    const std::vector<std::pair<float, float>> IOU = {std::make_pair(0.0f, 1.0f), std::make_pair(0.1f, 1.0f), std::make_pair(0.3f, 1.0f),
                                            std::make_pair(0.5f, 1.0f), std::make_pair(0.45f, 1.0f), std::make_pair(0.35f, 1.0f), std::make_pair(0.0f, 1.0f) };   
    int sample_option;
    std::pair<float, float> iou;
    float min_iou, max_iou;
    in_width = _crop_param->in_width;
    in_height = _crop_param->in_height;
    bool invalid_bboxes = true;
    _entire_iou = true;
    BoundingBoxCord crop_box, jth_box;
    Parameter<float> * x_drift_factor = _crop_param->get_x_drift_factor();
    Parameter<float> * y_drift_factor = _crop_param->get_y_drift_factor();
    _x1_val = _crop_param->get_x1_arr_val();
    _y1_val = _crop_param->get_y1_arr_val();
    _crop_width_val = _crop_param->get_cropw_arr_val();
    _crop_height_val = _crop_param->get_croph_arr_val();
    for (uint i = 0; i < _batch_size; i++)
    {
        int bb_count = _meta_data_info->get_bb_labels_batch()[i].size();
        std::vector<int> labels_buf(bb_count);
        memcpy(labels_buf.data(), _meta_data_info->get_bb_labels_batch()[i].data(), sizeof(int) * bb_count);
        std::vector<float> coords_buf(bb_count * 4);
        memcpy(coords_buf.data(), _meta_data_info->get_bb_cords_batch()[i].data(), _meta_data_info->get_bb_cords_batch()[i].size() * sizeof(BoundingBoxCord));
        crop_box.h = _crop_height_val[i];
        crop_box.w = _crop_width_val[i];
        crop_box.x = _x1_val[i];
        crop_box.y = _y1_val[i];
        while (true)
        {
            sample_option = dis(gen);
            iou =  IOU[sample_option];
            _iou_range[i] = iou;
            min_iou = iou.first;
            max_iou = iou.second;
            //sample_option = 0;
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
            if ((crop_box.w / crop_box.h < 0.5) || (crop_box.w / crop_box.h > 2.))
                continue;
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
                float bb_iou = ssd_BBoxIntersectionOverUnion(jth_box, crop_box, _entire_iou); 
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
        _x1_val[i] = crop_box.x;
        _y1_val[i] = crop_box.y;
        _crop_width_val[i] = crop_box.w;
        _crop_height_val[i] = crop_box.h;
        _x2_val[i] = _x1_val[i] + crop_box.w;
        _y2_val[i] = _y2_val[i] + crop_box.h;

    }
    vxCopyArrayRange((vx_array)_crop_param->cropw_arr, 0, _batch_size, sizeof(uint), _crop_width_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_crop_param->croph_arr, 0, _batch_size, sizeof(uint), _crop_height_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_crop_param->x1_arr, 0, _batch_size, sizeof(uint), _x1_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    vxCopyArrayRange((vx_array)_crop_param->y1_arr, 0, _batch_size, sizeof(uint), _y1_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    // std::cerr<<"\n Crop values after ssd ::";
    // for(int i = 0; i < _batch_size; i++)
    // {
    //     std::cerr<<"\n "<<_x1_val[i]<<"\t"<<_y1_val[i]<<"\t"<<_crop_width_val[i]<<"\t"<<_crop_height_val[i];
    // }
    _outputs[0]->update_image_roi(_crop_width_val, _crop_height_val);  
}

void SSDRandomCropNode::init(FloatParam *crop_area_factor, FloatParam *crop_aspect_ratio, FloatParam *x_drift, FloatParam *y_drift, int num_of_attempts)
{
    _crop_param->set_x_drift_factor(core(x_drift));
    _crop_param->set_y_drift_factor(core(y_drift));
    _crop_param->set_area_factor(core(crop_area_factor));
    _crop_param->set_aspect_ratio(core(crop_aspect_ratio));
    _num_of_attempts = num_of_attempts;
}

#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_crop_factory.h"

class SSDRandomCropNode : public Node
{
public:
    SSDRandomCropNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    SSDRandomCropNode() = delete;
    void init(FloatParam *crop_area_factor, FloatParam *crop_aspect_ratio, FloatParam *x_drift, FloatParam *y_drift, int num_of_attempts);
    unsigned int get_dst_width() { return _outputs[0]->info().width(); }
    unsigned int get_dst_height() { return _outputs[0]->info().height_single(); }
    std::shared_ptr<RaliRandomCropParam> get_crop_param() { return _crop_param; }
    float get_threshold(){return _threshold;}
    std::vector<std::pair<float,float>> get_iou_range(){return _iou_range;}
    bool is_entire_iou(){return _entire_iou;}
    void set_meta_data_batch() {}

protected:
    void create_node() override;
    void update_node() override;

private:
    std::shared_ptr<RaliRandomCropParam> _meta_crop_param;
    vx_array _crop_width, _crop_height, _x1, _y1, _x2, _y2;
    std::vector<uint> _crop_width_val, _crop_height_val, _x1_val, _y1_val, _x2_val, _y2_val;
    // unsigned int _dst_width, _dst_height;
    std::vector<uint32_t> in_width, in_height;
    size_t _dest_width;
    size_t _dest_height;
    float  _threshold = 0.05;
    std::vector<std::pair<float,float>> _iou_range;
    int _num_of_attempts = 20;
    bool _entire_iou = false;
    std::shared_ptr<RaliRandomCropParam> _crop_param;
};
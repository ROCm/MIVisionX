#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_crop.h"

class CropParam;

class CropNode : public Node
{
public:
    CropNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    CropNode() = delete;
    void create(std::shared_ptr<Graph> graph) override;
    void init(unsigned int crop_h, unsigned int crop_w, float x_drift, float y_drift);
    void init(unsigned int crop_h, unsigned int crop_w);
    void init( FloatParam *crop_h_factor, FloatParam *crop_w_factor, FloatParam * x_drift, FloatParam * y_drift);
    void update_parameters() override;
private:

    size_t _dest_width;
    size_t _dest_height;
    std::shared_ptr<CropParam> _crop_param;
    vx_array src_width_array,src_height_array ,dst_width_array ,dst_height_array;
    std::vector<vx_uint32> src_width ,src_height, dst_width, dst_height;
    void update_dimensions();
};


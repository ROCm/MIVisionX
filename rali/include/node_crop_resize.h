#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_crop_resize.h"

class RandomCropResizeParam;

class CropResizeNode : public Node
{
public:
    CropResizeNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    CropResizeNode() = delete;
    void create(std::shared_ptr<Graph> graph) override;
    void init(float area, float aspect_ratio, float x_center_drift, float y_center_drift);
    void init(FloatParam* area, FloatParam *aspect_ratio, FloatParam * x_center_drift, FloatParam * y_center_drift);

    void update_parameters() override;
private:

    size_t _dest_width;
    size_t _dest_height;
    std::shared_ptr<RandomCropResizeParam> _crop_param;

    vx_array src_width_array,src_height_array ,dst_width_array ,dst_height_array;
    std::vector<vx_uint32> src_width ,src_height, dst_width, dst_height;
    void update_dimensions();
};




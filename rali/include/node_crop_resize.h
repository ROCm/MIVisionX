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
    void init(float area, float aspect_ratio, float x_center_drift, float y_center_drift);
    void init(FloatParam* area, FloatParam *aspect_ratio, FloatParam * x_center_drift, FloatParam * y_center_drift);

protected:
    void create_node() override;
    void update_node() override;
private:

    size_t _dest_width;
    size_t _dest_height;
    std::shared_ptr<RandomCropResizeParam> _crop_param;
    vx_array _dst_roi_width ,_dst_roi_height;
};




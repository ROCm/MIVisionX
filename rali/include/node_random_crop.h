#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_crop_factory.h"
#include "parameter_rali_crop.h"

class RandomCropNode : public Node
{
public:
    RandomCropNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    RandomCropNode() = delete;
    void init(float area , float aspect_ratio, float x_drift, float y_drift);
    void init( FloatParam *crop_area_factor, FloatParam *crop_aspect_ratio, FloatParam * x_drift, FloatParam * y_drift);
protected:
    void create_node() override ;
    void update_node() override;
private:
    size_t _dest_width;
    size_t _dest_height;
    std::shared_ptr<RaliRandomCropParam> _crop_param;
};


#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_crop_factory.h"
#include "parameter_rali_crop.h"

class CropNode : public Node
{
public:
    CropNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    CropNode() = delete;
    void init(unsigned int crop_h, unsigned int crop_w, float x_drift, float y_drift);
    void init(unsigned int crop_h, unsigned int crop_w);
    void init( FloatParam *crop_h_factor, FloatParam *crop_w_factor, FloatParam * x_drift, FloatParam * y_drift);
protected:
    void create_node() override ;
    void update_node() override;
private:
    size_t _dest_width;
    size_t _dest_height;
    std::shared_ptr<RaliCropParam> _crop_param;
};


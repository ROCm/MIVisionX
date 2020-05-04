#pragma once
#include "node.h"
#include "parameter_vx.h"
#include "parameter_factory.h"
#include "parameter_crop.h"

class CropParam;

class ResizeCropMirrorNode : public Node
{
public:
    ResizeCropMirrorNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    ResizeCropMirrorNode() = delete;
    void init(unsigned int crop_h, unsigned int crop_w, IntParam *mirror);
    void init( FloatParam *crop_h_factor, FloatParam *crop_w_factor, IntParam *mirror);
protected:
    void create_node() override;
    void update_node() override;
private:
    std::shared_ptr<CropParam> _crop_param;
    vx_array _dst_roi_width ,_dst_roi_height;
    ParameterVX<int> _mirror; 
    constexpr static int MIRROR_RANGE [2] =  {0, 1};
};


#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_crop_factory.h"
#include "parameter_vx.h"
class CropMirrorNormalizeNode : public Node 
{
public:
    CropMirrorNormalizeNode(const std::vector<Image *> &inputs,
                            const std::vector<Image *> &outputs);
    CropMirrorNormalizeNode() = delete;
    void init(int crop_h, int crop_w, float start_x, float start_y, float mean, float std_dev, IntParam *mirror);
protected:
    void create_node() override ;
    void update_node() override;
private:
    std::shared_ptr<RaliCropParam> _crop_param;
    vx_array _src_width_array, _src_height_array;
    std::vector<vx_float32> _mean_vx, _std_dev_vx;
    vx_array _mean_array, _std_dev_array;
    float _mean; 
    float _std_dev; 
    ParameterVX<int> _mirror;
    constexpr static int   MIRROR_RANGE [2] =  {0, 1};
};
#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"
#include "parameter_crop_resize.h"

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
    vx_array _src_width_array, _src_height_array;
    std::vector<vx_uint32> _x1, _x2, _y1, _y2;
    std::vector<vx_float32> _mean_vx, _std_dev_vx;
    vx_array _mean_array, _std_dev_array;
    vx_array _x1_array, _x2_array, _y1_array, _y2_array;
    int _crop_h;
    int _crop_w;
    int _crop_d;
    float _mean; // vector of means in future 
    float _std_dev; // vector of std_devs in future
    ParameterVX<int> _mirror; // Should come from int random number generator with values 1 or 0 - Coin Flip
    constexpr static int   MIRROR_RANGE [2] =  {0, 1};
};
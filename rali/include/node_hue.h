#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"


class HueNode : public Node
{
public:
    void create(std::shared_ptr<Graph> graph) override;
    HueNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    HueNode() = delete;
    void init(float hue);
    void init(FloatParam *hue);
    void update_parameters() override;

private:
    ParameterVX<float> _hue;
    std::vector<vx_uint32> _width, _height;
    vx_array _width_array,_height_array;
    constexpr static float HUE_RANGE [2] = {-359.0, 359.0};
    void update_dimensions();
};

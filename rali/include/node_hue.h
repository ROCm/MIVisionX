#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"


class HueNode : public Node
{
public:
    HueNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    HueNode() = delete;
    void init(float hue);
    void init(FloatParam *hue);
protected:
    void create_node() override;
    void update_node() override;
private:
    ParameterVX<float> _hue;
    constexpr static float HUE_RANGE [2] = {-359.0, 359.0};
};

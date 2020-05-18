#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"
class FogNode : public Node
{
public:
    FogNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    FogNode() = delete;
    void init(float fog_param);
    void init(FloatParam *fog_param);
protected:
    void create_node() override;
    void update_node() override;
private:
    ParameterVX<float> _fog_param;
    constexpr static float FOG_VALUE_RANGE [2] = {0.2, 0.8};
};



#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"




class FogNode : public Node
{
public:
    void create(std::shared_ptr<Graph> graph) override;
    FogNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs);
    FogNode() = delete;
    void init(float shift);
    void init(FloatParam *shift);
    void update_parameters() override;

private:
    ParameterVX<float> _fog_param;
    constexpr static unsigned FOG_VALUE_OVX_PARAM_IDX = 2;
    constexpr static float FOG_VALUE_RANGE [2] = {0.2, 0.8};
};



#pragma once

#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"

class SnowNode : public Node
{
public:
    void create(std::shared_ptr<Graph> graph) override;
    SnowNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs);
    SnowNode() = delete;
    void init(float shfit);
    void init(FloatParam *shift);
    void update_parameters() override;

private:
    ParameterVX<float> _shift;
    constexpr static float SNOW_VALUE_RANGE [2] = {0.1, 0.8};
    constexpr static unsigned SNOW_VALUE_OVX_PARAM_IDX = 2;
};

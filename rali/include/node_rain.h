#pragma once

#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"

class RainNode : public Node
{
public:
    void create(std::shared_ptr<Graph> graph) override;
    RainNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs);
    RainNode() = delete;
    void init(float rain_value);
    void init(FloatParam *rain_value);
    void update_parameters() override;

private:
    ParameterVX<float> _shift;
    constexpr static float RAIN_VALUE_RANGE [2] = {0.15, 0.95};
    constexpr static unsigned RAIN_VALUE_OVX_PARAM_IDX = 2;

    constexpr static size_t RAIN_WIDTH = 1;
    constexpr static size_t RAIN_HEIGHT = 15;
    constexpr static float RAIN_TRANSPARENCY = 0.2;
};




#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"


class GammaNode : public Node
{
public:
    void create(std::shared_ptr<Graph> graph) override;
    GammaNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs);
    GammaNode() = delete;
    void init(float shift);
    void init(FloatParam *shift);
    void update_parameters() override;

private:
    ParameterVX<float> _shift;
    constexpr static float SHIFT_RANGE [2] = {0.3, 7.00};
    constexpr static unsigned SHIFT_OVX_PARAM_IDX = 2;
};

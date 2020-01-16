#pragma once

#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"
#include "graph.h"

class SnPNoiseNode : public Node
{
public:
    void create(std::shared_ptr<Graph> graph) override ;
    SnPNoiseNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs);
    SnPNoiseNode() = delete;
    void init(float sdev);
    void init(FloatParam *sdev);
    void update_parameters() override;

private:
    ParameterVX<float> _sdev;
    constexpr static float SDEV_RANGE [2] = {0.15, 0.6};
    constexpr static unsigned SDEV_OVX_PARAM_IDX = 2;
};


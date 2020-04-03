#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"
#include "graph.h"

class BlurNode : public Node
{
public:
    BlurNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs);
    BlurNode() = delete;
    void create(std::shared_ptr<Graph> graph) override ;
    void init( float sdev);
    void init( FloatParam *sdev);
    void update_parameters() override;

private:
    ParameterVX<float> _sdev;
    constexpr static float SDEV_RANGE [2] = {0.3, 5.0};
    constexpr static unsigned SDEV_OVX_PARAM_IDX = 2;
};

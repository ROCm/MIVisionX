#pragma once

#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"
#include "graph.h"

class SnPNoiseNode : public Node
{
public:
    SnPNoiseNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    SnPNoiseNode() = delete;
    void init(float sdev);
    void init(FloatParam *sdev);
protected:
    void create_node() override;
    void update_node() override;
private:
    ParameterVX<float> _sdev;
    constexpr static float SDEV_RANGE [2] = {0.1, 0.15};
};


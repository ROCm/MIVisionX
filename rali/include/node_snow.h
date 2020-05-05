#pragma once

#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"

class SnowNode : public Node
{
public:
    SnowNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    SnowNode() = delete;
    void init(float shift);
    void init(FloatParam *shift);
protected:
    void create_node() override;
    void update_node() override;
private:
    ParameterVX<float> _shift;
    constexpr static float SNOW_VALUE_RANGE [2] = {0.1, 0.8};
};

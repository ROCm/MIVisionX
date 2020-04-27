#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"


class SatNode : public Node
{
public:
    SatNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    SatNode() = delete;
    void init(float sat);
    void init(FloatParam *sat);
protected:
    void create_node() override;
    void update_node() override;
private:
    ParameterVX<float> _sat; // For saturation
    constexpr static float SAT_RANGE [2] = {-0.5, 0.5};
};

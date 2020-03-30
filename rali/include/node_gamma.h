#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"


class GammaNode : public Node
{
public:
    GammaNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    GammaNode() = delete;
    void init(float shift);
    void init(FloatParam *shift);

protected:
    void update_node() override;
    void create_node() override;
private:
    ParameterVX<float> _shift;
    constexpr static float SHIFT_RANGE [2] = {0.3, 7.00};
};

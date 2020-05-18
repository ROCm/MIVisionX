#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"
#include "graph.h"

class ColorTwistBatchNode : public Node
{
public:
    ColorTwistBatchNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    ColorTwistBatchNode() = delete;
    void init(float alpha, float beta, float hue, float sat);
    void init(FloatParam *alpha, FloatParam *beta, FloatParam *hue, FloatParam *sat);

protected:
    void create_node() override;
    void update_node() override;
private:

    ParameterVX<float> _alpha;
    ParameterVX<float> _beta;
    ParameterVX<float> _hue;
    ParameterVX<float> _sat;

    constexpr static float   ALPHA_RANGE [2] = {0.1, 1.95};
    constexpr static float   BETA_RANGE [2] = {0.1, 25.0};
    constexpr static float   HUE_RANGE [2] = {5.0, 170.0};
    constexpr static float   SAT_RANGE [2] = {0.1, 0.4};
};
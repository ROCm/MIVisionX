#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"
#include "graph.h"

class BrightnessNode : public Node
{
public:
    BrightnessNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    BrightnessNode() = delete;

    void init( float alpha, int beta);
    void init( FloatParam* alpha_param, IntParam* beta_param);

protected:
    void create_node() override ;
    void update_node() override;
private:

    ParameterVX<float> _alpha;
    ParameterVX<int> _beta;
    constexpr static float ALPHA_RANGE [2] = {0.1, 1.95};
    constexpr static int   BETA_RANGE [2] = {0, 25};
};
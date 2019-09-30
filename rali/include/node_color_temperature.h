#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"
#include "graph.h"

class ColorTemperatureNode : public Node
{
public:
    ColorTemperatureNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs);
    void create(std::shared_ptr<Graph> graph) override;
    void init(int shift);
    void init( IntParam *shift);
    ColorTemperatureNode() = delete;
    void update_parameters() override;

private:
    ParameterVX<int> _adj_value_param;
    constexpr static int ADJUSTMENT_RANGE [2] = {-99, 99};
    constexpr static unsigned ADJUSTMENT_OVX_PARAM_IDX = 2;
};
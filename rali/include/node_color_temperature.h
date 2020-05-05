#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"
#include "graph.h"

class ColorTemperatureNode : public Node
{
public:
    ColorTemperatureNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);

    ColorTemperatureNode() = delete;
    void init(int adjustment);
    void init(IntParam *adjustment);

protected:
    void create_node() override ;
    void update_node() override;
private:
    ParameterVX<int> _adj_value_param;
    constexpr static int ADJUSTMENT_RANGE [2] = {-99, 99};
};
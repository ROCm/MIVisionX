#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"
#include "graph.h"

class ColorTemperatureNode : public Node
{
public:
    ColorTemperatureNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    void create(std::shared_ptr<Graph> graph) override;
    ColorTemperatureNode() = delete;
    void init(int adjustment);
    void init(IntParam *adjustment);
    void update_parameters() override;

private:
    ParameterVX<int> _adj_value_param;
    constexpr static int ADJUSTMENT_RANGE [2] = {-99, 99};
    std::vector<vx_uint32> _width,_height;
    vx_array  _width_array ,_height_array;
    void update_dimensions();
};
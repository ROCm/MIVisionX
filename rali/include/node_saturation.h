#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"


class SatNode : public Node
{
public:
    void create(std::shared_ptr<Graph> graph) override;
    SatNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    SatNode() = delete;
    void init(float sat);
    void init(FloatParam *sat);
    void update_parameters() override;

private:
    ParameterVX<float> _sat; // For saturation
    std::vector<vx_uint32> _width, _height;
    vx_array _width_array,_height_array;
    constexpr static float SAT_RANGE [2] = {-0.5, 0.5};
    void update_dimensions();
};

#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"
#include "graph.h"

class ExposureNode : public Node
{
public:
    void create(std::shared_ptr<Graph> graph) override ;
    ExposureNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    ExposureNode() = delete;
    void init(float shift);
    void init(FloatParam *shift);
    void update_parameters() override;

private:
    ParameterVX<float> _shift;
    std::vector<vx_uint32> _width, _height;
    vx_array _width_array ,_height_array;
    void update_dimensions();
    constexpr static float SHIFT_RANGE [2] = {0.15, 0.95};
};
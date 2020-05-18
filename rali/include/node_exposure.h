#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"
#include "graph.h"

class ExposureNode : public Node
{
public:
    ExposureNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    ExposureNode() = delete;
    void init(float shift);
    void init(FloatParam *shift);
protected:
    void create_node() override;
    void update_node() override;
private:
    ParameterVX<float> _shift;
    vx_array _width_array ,_height_array;
    constexpr static float SHIFT_RANGE [2] = {0.15, 0.95};
};
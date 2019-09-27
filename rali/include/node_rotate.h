#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"
#include "graph.h"

class RotateNode : public Node
{
public:
    RotateNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs);
    RotateNode() = delete;
    void create(std::shared_ptr<Graph> graph) override;
    void init(FloatParam *angle);
    void init(float angle);
    void update_parameters() override;
private:
    ParameterVX<float> _angle;
    constexpr static unsigned ROTATE_ANGLE_OVX_PARAM_IDX = 4;
    constexpr static float ROTATE_ANGLE_RANGE [2] = {0, 180};

};


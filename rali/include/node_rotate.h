#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"
#include "graph.h"

class RotateNode : public Node
{
public:
    RotateNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    RotateNode() = delete;
    void init(float angle);
    void init(FloatParam *angle);

protected:
    void create_node() override;
    void update_node() override;
private:
    ParameterVX<float> _angle;
    vx_array _dst_roi_width,_dst_roi_height;
    constexpr static float ROTATE_ANGLE_RANGE [2] = {0, 180};

};
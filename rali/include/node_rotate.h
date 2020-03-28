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
    void create(std::shared_ptr<Graph> graph) override;
    void init(float angle);
    void init(FloatParam *angle);
    void update_parameters() override;
private:
    ParameterVX<float> _angle;
    std::vector<vx_uint32> _width, _height, dst_width, dst_height;
   // vx_uint32 *_width, *_height;
    vx_array _width_array,_height_array,dst_width_array,dst_height_array;
    constexpr static float ROTATE_ANGLE_RANGE [2] = {0, 180};
    void update_dimensions();

};
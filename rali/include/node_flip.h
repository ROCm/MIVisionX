#pragma once
#include "node.h"
#include "parameter_vx.h"
#include "parameter_factory.h"

class FlipNode : public Node
{
public:
    void create(std::shared_ptr<Graph> graph) override ;
    FlipNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    FlipNode() = delete;
    void init(int flip_axis);
    void init(IntParam *flip_axis);
    void update_parameters() override;

private:
    int _axis;
    ParameterVX<int> _flip_axis;
    constexpr static int   FLIP_SIZE [2] =  {0, 2};

    std::vector<vx_uint32> _width, _height;
    vx_array _width_array ,_height_array;
    void update_dimensions();
};
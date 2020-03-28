#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_vx.h"
#include "graph.h"

class BlurNode : public Node
{
public:
    BlurNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    BlurNode() = delete;
    void create(std::shared_ptr<Graph> graph) override ;
    void init(int sdev);
    void init(IntParam *sdev);
    void update_parameters() override;

private:
    ParameterVX<int> _sdev;
    constexpr static int SDEV_RANGE [2] = {3, 9};
    std::vector<vx_uint32> _width, _height;
    vx_array _width_array ,_height_array;
    void update_dimensions();
};

#pragma once
#include "node.h"
#include "parameter_vx.h"
#include "parameter_factory.h"

class FlipNode : public Node
{
public:
    FlipNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    FlipNode() = delete;
    void init(int flip_axis);
    void init(IntParam *flip_axis);
protected:
    void create_node() override;
    void update_node() override;
private:
    int _axis;
    ParameterVX<int> _flip_axis;
    constexpr static int   FLIP_SIZE [2] =  {0, 2};
};
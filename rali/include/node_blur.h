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
    void init(int sdev);
    void init(IntParam *sdev);

protected:
    void update_node() override;
    void create_node() override;

private:
    ParameterVX<int> _sdev;
    constexpr static int SDEV_RANGE [2] = {3, 9};
};

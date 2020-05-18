#pragma once

#include "node.h"
#include "parameter_factory.h"
#include "graph.h"


class PixelateNode : public Node
{
public:
    PixelateNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    PixelateNode() = delete;
protected:
    void create_node() override;
    void update_node() override;
private:
};

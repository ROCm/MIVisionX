#pragma once
#include "node.h"
#include "graph.h"

class NopNode : public Node
{
public:
    NopNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    NopNode() = delete;
protected:
    void create_node() override;
    void update_node() override;
};

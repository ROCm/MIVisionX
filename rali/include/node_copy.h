#pragma once
#include "node.h"
#include "graph.h"

class CopyNode : public Node
{
public:
    CopyNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    CopyNode() = delete;

protected:
    void create_node() override;
    void update_node() override {};
};

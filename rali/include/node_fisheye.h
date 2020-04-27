#pragma once

#include "node.h"
#include "graph.h"


class FisheyeNode : public Node
{
public:
    FisheyeNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs);
    FisheyeNode() = delete;

protected:
    void create_node() override;
    void update_node() override;
private:
};

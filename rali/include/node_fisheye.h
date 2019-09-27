#pragma once

#include "node.h"
#include "graph.h"


class FisheyeNode : public Node
{
public:
    void create(std::shared_ptr<Graph> graph) override;
    FisheyeNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs);
    FisheyeNode() = delete;
    void update_parameters() override;
};
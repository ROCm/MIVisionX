#pragma once
#include "node.h"
#include "graph.h"

class NopNode : public Node
{
public:
    NopNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs);
    NopNode() = delete;
    void create(std::shared_ptr<Graph> graph) override ;
    void update_parameters() override;
};

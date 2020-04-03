#pragma once
#include "node.h"
#include "graph.h"

class CopyNode : public Node
{
public:
    CopyNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs);
    CopyNode() = delete;
    void create(std::shared_ptr<Graph> graph) override ;
    void update_parameters() override;
};

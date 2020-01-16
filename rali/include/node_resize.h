#pragma once
#include "node.h"

class ResizeNode : public Node
{
public:
    void create(std::shared_ptr<Graph> graph) override ;
    ResizeNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs);
    ResizeNode() = delete;
    void update_parameters() override;
};

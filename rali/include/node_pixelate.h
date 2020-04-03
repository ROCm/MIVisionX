#pragma once

#include "node.h"
#include "parameter_factory.h"
#include "graph.h"


class PixelateNode : public Node
{
public:
    void create(std::shared_ptr<Graph> graph) override;
    PixelateNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs);
    PixelateNode() = delete;
    void update_parameters() override;
};

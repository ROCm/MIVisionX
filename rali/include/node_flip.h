#pragma once
#include "node.h"
#include "parameter_factory.h"

class FlipNode : public Node
{
public:
    void create(std::shared_ptr<Graph> graph) override ;
    FlipNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs);
    FlipNode() = delete;
    void init( int axis);
    void update_parameters() override;

private:
    int _axis;
};
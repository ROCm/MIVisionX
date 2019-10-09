#pragma once
#include <set>
#include <memory>
#include "graph.h"
#include "image.h"
class Node
{
public:
    Node(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs):
        _inputs(inputs),
        _outputs(outputs)
    {}
    virtual ~Node() {
        //_next_nodes.clear();
        //_prev_nodes.clear();
        if(_node)
            vxReleaseNode(&_node);
    }
    virtual void create(std::shared_ptr<Graph> graph) = 0;
    virtual void update_parameters() = 0;
    std::vector<Image*> input() { return _inputs; };
    std::vector<Image*> output() { return _outputs; };
    void add_next(std::shared_ptr<Node> node) { return; }//_next_nodes.insert(node); }
    void add_previous(std::shared_ptr<Node> node) { return; }//_prev_nodes.insert(node); }
   // std::set<std::shared_ptr<Node>> get_prev() { return _prev_nodes; }
   // std::set<std::shared_ptr<Node>> get_next() { return _next_nodes; }
    std::shared_ptr<Graph> graph() { return _graph; }
protected:
    //std::set<std::shared_ptr<Node>> _next_nodes;
    //std::set<std::shared_ptr<Node>> _prev_nodes;
    std::vector<Image*> _inputs;
    std::vector<Image*> _outputs;
    std::shared_ptr<Graph> _graph = nullptr;
    vx_node _node = nullptr;
};

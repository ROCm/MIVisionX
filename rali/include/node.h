#pragma once
#include <set>
#include <memory>
#include "graph.h"
#include "image.h"
class Node
{
public:
    Node(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        _inputs(inputs),
        _outputs(outputs),
        _batch_size(outputs[0]->info().batch_size()) {}
    virtual ~Node();
    void create(std::shared_ptr<Graph> graph);
    void update_parameters();
    std::vector<Image*> input() { return _inputs; };
    std::vector<Image*> output() { return _outputs; };
    void add_next(const std::shared_ptr<Node>& node) {} // To be implemented
    void add_previous(const std::shared_ptr<Node>& node) {} //To be implemented
    std::shared_ptr<Graph> graph() { return _graph; }
protected:
    virtual void create_node() = 0;
    virtual void update_node() = 0;
    virtual void update_src_roi();
    std::vector<Image*> _inputs;
    std::vector<Image*> _outputs;
    std::shared_ptr<Graph> _graph = nullptr;
    vx_array _src_roi_width = nullptr;
    vx_array _src_roi_height = nullptr;
    vx_node _node = nullptr;
    size_t _batch_size;
};

#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_flip.h"
#include "exception.h"

FlipNode::FlipNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs):
        Node(inputs, outputs),
        _axis(0)
{
}

void FlipNode::create(std::shared_ptr<Graph> graph)
{
    if(_node)
        return;

    _graph = graph;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")

    _node = vxExtrppNode_Flip(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _axis);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the flip (vxExtrppNode_Flip) node failed: "+ TOSTR(status))

}

void FlipNode::init(int axis)
{
    _axis = axis;
}

void FlipNode::update_parameters()
{
}

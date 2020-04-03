#include <vx_ext_rpp.h>
#include "node_fisheye.h"
#include "exception.h"

FisheyeNode::FisheyeNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs):
        Node(inputs, outputs)
{
}

void FisheyeNode::create(std::shared_ptr<Graph> graph)
{
    if(_node)
        return;

    _graph = graph;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")

    _node = vxExtrppNode_Fisheye(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle());

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the fish eye (vxExtrppNode_Fisheye) node failed: "+ TOSTR(status))


}

void FisheyeNode::update_parameters()
{
}

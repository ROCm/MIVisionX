#include <vx_ext_rpp.h>
#include "node_nop.h"
#include "exception.h"

NopNode::NopNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs)
{
}

void NopNode::create_node()
{
    if(_node)
        return;


    _node = vxExtrppNode_Nop(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle());

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the nop (vxNopNode) node failed: "+ TOSTR(status))

}

void NopNode::update_node()
{
}
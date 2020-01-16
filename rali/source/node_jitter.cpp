#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_jitter.h"
#include "exception.h"


JitterNode::JitterNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs):
        Node(inputs, outputs),
        _kernel_size(KERNEL_SIZE_OVX_PARAM_IDX, KERNEL_SIZE[0], KERNEL_SIZE[1])
{
}

void JitterNode::create(std::shared_ptr<Graph> graph)
{
    if(_node)
        return;

    _graph = graph;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")

    _node = vxExtrppNode_Jitter(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _kernel_size.default_value());

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the jitter (vxExtrppNode_Jitter) node failed: "+ TOSTR(status))

    _kernel_size.create(_node);
}

void JitterNode::init(int kernel_size)
{
    _kernel_size.set_param(kernel_size);
}

void JitterNode::init(IntParam *kernel_size)
{
    _kernel_size.set_param(core(kernel_size));
}

void JitterNode::update_parameters()
{
    _kernel_size.update();
}


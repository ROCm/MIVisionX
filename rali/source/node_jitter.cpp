#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_jitter.h"
#include "exception.h"


JitterNode::JitterNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _kernel_size(KERNEL_SIZE[0], KERNEL_SIZE[1])
{
}

void JitterNode::create_node()
{
    if(_node)
        return;

    _kernel_size.create_array(_graph ,VX_TYPE_UINT32, _batch_size);
    _node = vxExtrppNode_JitterbatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _kernel_size.default_array(), _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the jitter (vxExtrppNode_Jitter) node failed: "+ TOSTR(status))
}

void JitterNode::init(int kernel_size)
{
    _kernel_size.set_param(kernel_size);
}

void JitterNode::init(IntParam *kernel_size)
{
    _kernel_size.set_param(core(kernel_size));
}

void JitterNode::update_node()
{
    _kernel_size.update_array();
}


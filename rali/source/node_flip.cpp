#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_flip.h"
#include "exception.h"

FlipNode::FlipNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _flip_axis(FLIP_SIZE[0], FLIP_SIZE[1])
{
}

void FlipNode::create_node()
{
    if(_node)
        return;


    _flip_axis.create_array(_graph ,VX_TYPE_UINT32 ,_batch_size);
    _node = vxExtrppNode_FlipbatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _flip_axis.default_array(), _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the flip (vxExtrppNode_Flip) node failed: "+ TOSTR(status))

}

void FlipNode::init(int flip_axis)
{
    _flip_axis.set_param(flip_axis);
}

void FlipNode::init(IntParam* flip_axis)
{
    _flip_axis.set_param(core(flip_axis));
}

void FlipNode::update_node()
{
    _flip_axis.update_array();
}

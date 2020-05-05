#include <vx_ext_rpp.h>
#include "node_blur.h"
#include "exception.h"

BlurNode::BlurNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _sdev(SDEV_RANGE[0], SDEV_RANGE[1])
{
}

void BlurNode::create_node()
{
    if(_node)
        return;

    _sdev.create_array(_graph ,VX_TYPE_UINT32, _batch_size);
    _node = vxExtrppNode_BlurbatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width,_src_roi_height, _outputs[0]->handle(), _sdev.default_array(), _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the blur (vxExtrppNode_blur) node failed: "+ TOSTR(status))

}

void BlurNode::init(int sdev)
{
    _sdev.set_param(sdev);
}

void BlurNode::init(IntParam* sdev)
{
    _sdev.set_param(core(sdev));
}

void BlurNode::update_node()
{
    _sdev.update_array();
}

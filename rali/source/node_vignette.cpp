#include <vx_ext_rpp.h>
#include <VX/vx_compatibility.h>
#include "node_vignette.h"
#include "exception.h"

VignetteNode::VignetteNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _sdev(SDEV_RANGE[0], SDEV_RANGE[1])
{
}

void VignetteNode::create_node()
{
    if(_node)
        return;

    _sdev.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);

    _node = vxExtrppNode_VignettebatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _sdev.default_array(), _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the vignette (vxExtrppNode_VignettebatchPD) node failed: "+ TOSTR(status))
}

void VignetteNode::init(float sdev)
{
    _sdev.set_param(sdev);
}

void VignetteNode::init(FloatParam* sdev)
{
    _sdev.set_param(core(sdev));
}

void VignetteNode::update_node()
{
    _sdev.update_array();
}


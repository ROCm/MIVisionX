#include <vx_ext_rpp.h>
#include "node_exposure.h"
#include "exception.h"

ExposureNode::ExposureNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _shift(SHIFT_RANGE[0], SHIFT_RANGE[1])
{
}

void ExposureNode::create_node()
{
    if(_node)
        return;

    _shift.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _node = vxExtrppNode_ExposurebatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _shift.default_array(), _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the exposure (vxExtrppNode_Exposure) node failed: "+ TOSTR(status))

}

void ExposureNode::init(float shfit)
{
    _shift.set_param(shfit);
}

void ExposureNode::init(FloatParam* shfit)
{
    _shift.set_param(core(shfit));
}

void ExposureNode::update_node()
{
    _shift.update_array();
}
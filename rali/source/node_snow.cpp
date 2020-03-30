#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_snow.h"
#include "exception.h"


SnowNode::SnowNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _shift(SNOW_VALUE_RANGE[0], SNOW_VALUE_RANGE[1])
{
}

void SnowNode::create_node()
{
    if(_node)
        return;

    _shift.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _node = vxExtrppNode_SnowbatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _shift.default_array(), _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the snow (vxExtrppNode_Snow) node failed: "+ TOSTR(status))
}

void SnowNode::init(float shfit)
{
    _shift.set_param(shfit);
}

void SnowNode::init(FloatParam* shfit)
{
    _shift.set_param(core(shfit));
}

void SnowNode::update_node()
{
    _shift.update_array();
}

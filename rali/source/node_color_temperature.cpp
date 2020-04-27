#include <vx_ext_rpp.h>
#include <VX/vx_compatibility.h>
#include "node_color_temperature.h"
#include "exception.h"

ColorTemperatureNode::ColorTemperatureNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _adj_value_param(ADJUSTMENT_RANGE[0], ADJUSTMENT_RANGE[1])
{
}

void ColorTemperatureNode::create_node()
{
    if(_node)
        return;

    _adj_value_param.create_array(_graph , VX_TYPE_INT32, _batch_size);

    _node = vxExtrppNode_ColorTemperaturebatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _adj_value_param.default_array(), _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the color temp batch (vxExtrppNode_ColorTemperaturebatchPD) node failed: "+ TOSTR(status))

}

void ColorTemperatureNode::init(int adjustment)
{
    _adj_value_param.set_param(adjustment);
}

void ColorTemperatureNode::init(IntParam* adjustment)
{
    _adj_value_param.set_param(core(adjustment));
}

void ColorTemperatureNode::update_node()
{
    _adj_value_param.update_array();
}


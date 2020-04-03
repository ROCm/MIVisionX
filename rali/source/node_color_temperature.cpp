#include <vx_ext_rpp.h>
#include <VX/vx_compatibility.h>
#include "node_color_temperature.h"
#include "exception.h"

ColorTemperatureNode::ColorTemperatureNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs):
        Node(inputs, outputs),
        _adj_value_param(ADJUSTMENT_OVX_PARAM_IDX, ADJUSTMENT_RANGE[0], ADJUSTMENT_RANGE[1])
{
}

void ColorTemperatureNode::create(std::shared_ptr<Graph> graph)
{
    if(_node)
        return;

    _graph = graph;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")

    _node = vxExtrppNode_ColorTemperature(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _adj_value_param.default_value());

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the color temp (vxExtrppNode_ColorTemperature) node failed: "+ TOSTR(status))

    _adj_value_param.create(_node);

}

void ColorTemperatureNode::init(int adj)
{
    _adj_value_param.set_param(adj);
}

void ColorTemperatureNode::init(IntParam* adj)
{
    _adj_value_param.set_param(core(adj));
}

void ColorTemperatureNode::update_parameters()
{
    _adj_value_param.update();
}


#include <vx_ext_rpp.h>
#include <VX/vx_compatibility.h>
#include <graph.h>
#include "node_snow.h"
#include "exception.h"


SnowNode::SnowNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs):
        Node(inputs, outputs),
        _shift(SNOW_VALUE_OVX_PARAM_IDX, SNOW_VALUE_RANGE[0], SNOW_VALUE_RANGE[1])
{
}

void SnowNode::create(std::shared_ptr<Graph> graph)
{
    if(_node)
        return;

    _graph = graph;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")

    _node = vxExtrppNode_Snow(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _shift.default_value());

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the snow (vxExtrppNode_Snow) node failed: "+ TOSTR(status))

    _shift.create(_node);

}

void SnowNode::init(float shfit)
{
    _shift.set_param(shfit);
}

void SnowNode::init(FloatParam* shfit)
{
    _shift.set_param(core(shfit));
}

void SnowNode::update_parameters()
{
    _shift.update();
}

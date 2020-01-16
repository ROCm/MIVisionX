#include <vx_ext_rpp.h>
#include <VX/vx_compatibility.h>
#include <graph.h>
#include "node_rain.h"
#include "exception.h"

RainNode::RainNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs):
        Node(inputs, outputs),
        _shift(RAIN_VALUE_OVX_PARAM_IDX, RAIN_VALUE_RANGE[0], RAIN_VALUE_RANGE[1])
{
}

void RainNode::create(std::shared_ptr<Graph> graph)
{
    if(_node)
        return;

    _graph = graph;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")

    _node = vxExtrppNode_Rain(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _shift.default_value(), RAIN_WIDTH, RAIN_HEIGHT, RAIN_TRANSPARENCY);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the rain (vxExtrppNode_Rain) node failed: "+ TOSTR(status))

    _shift.create(_node);

}

void RainNode::init(float shfit)
{
    _shift.set_param(shfit);
}

void RainNode::init(FloatParam* shfit)
{
    _shift.set_param(core(shfit));
}

void RainNode::update_parameters()
{
    _shift.update();
}

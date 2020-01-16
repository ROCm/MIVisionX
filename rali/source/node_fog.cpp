#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_fog.h"
#include "exception.h"

FogNode::FogNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs):
        Node(inputs, outputs),
        _fog_param(FOG_VALUE_OVX_PARAM_IDX, FOG_VALUE_RANGE[0], FOG_VALUE_RANGE[1])
{
}

void FogNode::create(std::shared_ptr<Graph> graph)
{
    if(_node)
        return;

    _graph = graph;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")

    _node = vxExtrppNode_Fog(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _fog_param.default_value());

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the fog (vxExtrppNode_Fog) node failed: "+ TOSTR(status))

    _fog_param.create(_node);

}

void FogNode::init(float shfit)
{
    _fog_param.set_param(shfit);
}

void FogNode::init(FloatParam* shfit)
{
    _fog_param.set_param(core(shfit));
}

void FogNode::update_parameters()
{
    _fog_param.update();
}

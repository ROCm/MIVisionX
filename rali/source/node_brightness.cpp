#include <vx_ext_rpp.h>
#include "node_brightness.h"
#include "exception.h"


BrightnessNode::BrightnessNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs):
        Node(inputs, outputs),
        _alpha(ALPHA_OVX_PARAM_IDX, ALPHA_RANGE[0], ALPHA_RANGE[1]),
        _beta (BETA_OVX_PARAM_IDX, BETA_RANGE[0], BETA_RANGE[1])
{
}

void BrightnessNode::init( float alpha, int beta)
{
    _alpha.set_param(alpha);
    _beta.set_param(beta);
}

void BrightnessNode::init( FloatParam* alpha, IntParam* beta)
{
    _alpha.set_param(core(alpha));
    _beta.set_param(core(beta));
}

void BrightnessNode::create(std::shared_ptr<Graph> graph)
{
    if(_node)
        return;

    _graph = graph;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")

    _node = vxExtrppNode_brightness(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _alpha.default_value(), _beta.default_value());

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the brightness (vxExtrppNode_brightness) node failed: "+ TOSTR(status))

    _alpha.create(_node);
    _beta.create(_node);

}


void BrightnessNode::update_parameters()
{
    _alpha.update();
    _beta.update();
}


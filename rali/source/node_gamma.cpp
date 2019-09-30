#include <vx_ext_rpp.h>
#include <VX/vx_compatibility.h>
#include <graph.h>
#include "node_gamma.h"
#include "exception.h"


GammaNode::GammaNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs):
        Node(inputs, outputs),
        _shift(SHIFT_OVX_PARAM_IDX, SHIFT_RANGE[0], SHIFT_RANGE[1])
{
}

void GammaNode::create(std::shared_ptr<Graph> graph)
{
    if(_node)
        return;

    _graph = graph;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")

    _node = vxExtrppNode_GammaCorrection(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _shift.default_value());

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the gamma (vxExtrppNode_GammaCorrection) node failed: "+ TOSTR(status))

    _shift.create(_node);

}

void GammaNode::init(float shfit)
{
    _shift.set_param(shfit);
}

void GammaNode::init(FloatParam* shfit)
{
    _shift.set_param(core(shfit));
}

void GammaNode::update_parameters()
{
    _shift.update();
}
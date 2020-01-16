#include <vx_ext_rpp.h>
#include "node_exposure.h"
#include "exception.h"

ExposureNode::ExposureNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs):
        Node(inputs, outputs),
        _shift(SHIFT_OVX_PARAM_IDX, SHIFT_RANGE[0], SHIFT_RANGE[1])
{
}

void ExposureNode::create(std::shared_ptr<Graph> graph)
{
    if(_node)
        return;

    _graph = graph;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")

    _node = vxExtrppNode_Exposure(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _shift.default_value());

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the exposure (vxExtrppNode_Exposure) node failed: "+ TOSTR(status))

    _shift.create(_node);

}

void ExposureNode::init(float shfit)
{
    _shift.set_param(shfit);
}

void ExposureNode::init(FloatParam* shfit)
{
    _shift.set_param(core(shfit));
}

void ExposureNode::update_parameters()
{
    _shift.update();
}

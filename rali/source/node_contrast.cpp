#include <vx_ext_rpp.h>
#include "node_contrast.h"
#include "exception.h"

RaliContrastNode::RaliContrastNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs):
        Node(inputs, outputs),
        _min(CONTRAST_MIN_OVX_PARAM_IDX, CONTRAST_MIN_RANGE[0], CONTRAST_MIN_RANGE[1]),
        _max(CONTRAST_MAX_OVX_PARAM_IDX, CONTRAST_MAX_RANGE[0], CONTRAST_MAX_RANGE[1])
{
}

void RaliContrastNode::create(std::shared_ptr<Graph> graph)
{
    if(_node)
        return;

    _graph = graph;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")

    _node = vxExtrppNode_contrast(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _min.default_value(), _max.default_value());

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the contrast (vxExtrppNode_contrast) node failed: "+ TOSTR(status))

    _min.create(_node);
    _max.create(_node);
}

void RaliContrastNode::init(int min, int max)
{
    _min.set_param(min);
    _max.set_param(max);
}

void RaliContrastNode::init(IntParam *min, IntParam* max)
{
    _min.set_param(core(min));
    _max.set_param(core(max));
}

void RaliContrastNode::update_parameters()
{
    _min.update();
    _max.update();
}


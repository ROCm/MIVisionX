#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_blend.h"
#include "exception.h"


BlendNode::BlendNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs):
        Node(inputs, outputs),
        _ratio(RATIO_OVX_PARAM_IDX, RATIO_RANGE[0], RATIO_RANGE[1])
{
}
void BlendNode::create(std::shared_ptr<Graph> graph)
{
    if(_node)
        return;

    _graph = graph;

    if(_inputs.size() < 2 || _outputs.empty())
        THROW("Uninitialized input/output arguments")

    _node = vxExtrppNode_Blend(_graph->get(), _inputs[0]->handle(), _inputs[1]->handle(), _outputs[0]->handle(), _ratio.default_value());

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the blend (vxExtrppNode_Blend) node failed: "+ TOSTR(status))

    _ratio.create(_node);

}

void BlendNode::init(float sdev)
{
    _ratio.set_param(sdev);
}

void BlendNode::init(FloatParam* sdev)
{
    _ratio.set_param(core(sdev));
}

void BlendNode::update_parameters()
{
    _ratio.update();
}
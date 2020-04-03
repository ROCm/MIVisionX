#include <vx_ext_rpp.h>
#include "node_blur.h"
#include "exception.h"

BlurNode::BlurNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs):
        Node(inputs, outputs),
        _sdev(SDEV_OVX_PARAM_IDX, SDEV_RANGE[0], SDEV_RANGE[1])
{
}

void BlurNode::create(std::shared_ptr<Graph> graph)
{
    if(_node)
        return;

    _graph = graph;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")

    _node = vxExtrppNode_blur(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _sdev.default_value());

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the blur (vxExtrppNode_blur) node failed: "+ TOSTR(status))

    _sdev.create(_node);

}

void BlurNode::init(float sdev)
{
    _sdev.set_param(sdev);
}

void BlurNode::init(FloatParam* sdev)
{
    _sdev.set_param(core(sdev));
}

void BlurNode::update_parameters()
{
    _sdev.update();
}

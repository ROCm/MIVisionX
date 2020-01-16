#include <vx_ext_rpp.h>
#include "node_rotate.h"
#include "exception.h"


RotateNode::RotateNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs):
        Node(inputs, outputs),
        _angle(ROTATE_ANGLE_OVX_PARAM_IDX, ROTATE_ANGLE_RANGE[0], ROTATE_ANGLE_RANGE[1])
{
}

void RotateNode::create(std::shared_ptr<Graph> graph)
{
    if(_node)
        return;

    _graph = graph;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")

    _node = vxExtrppNode_Rotate(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _outputs[0]->info().width(), _outputs[0]->info().height_batch(), _angle.default_value());

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the rotate (vxExtrppNode_Rotate) node failed: "+ TOSTR(status))

    _angle.create(_node);

}

void RotateNode::init(float shfit)
{
    _angle.set_param(shfit);
}

void RotateNode::init(FloatParam* shfit)
{
    _angle.set_param(core(shfit));
}

void RotateNode::update_parameters()
{
    _angle.update();
}

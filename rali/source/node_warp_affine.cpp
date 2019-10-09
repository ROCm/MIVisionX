#include <vx_ext_rpp.h>
#include <VX/vx_compatibility.h>
#include "node_warp_affine.h"
#include "exception.h"


WarpAffineNode::WarpAffineNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs):
        Node(inputs, outputs),
        _x0(COEFFICIENT_X0_OVX_PARAM_IDX, COEFFICIENT_RANGE_1[0], COEFFICIENT_RANGE_1[1]),
        _x1(COEFFICIENT_X1_OVX_PARAM_IDX, COEFFICIENT_RANGE_0[0], COEFFICIENT_RANGE_0[1]),
        _y0(COEFFICIENT_Y0_OVX_PARAM_IDX, COEFFICIENT_RANGE_0[0], COEFFICIENT_RANGE_0[1]),
        _y1(COEFFICIENT_Y1_OVX_PARAM_IDX, COEFFICIENT_RANGE_1[0], COEFFICIENT_RANGE_1[1]),
        _o0(COEFFICIENT_O0_OVX_PARAM_IDX, COEFFICIENT_RANGE_OFFSET[0], COEFFICIENT_RANGE_OFFSET[1]),
        _o1(COEFFICIENT_O1_OVX_PARAM_IDX, COEFFICIENT_RANGE_OFFSET[0], COEFFICIENT_RANGE_OFFSET[1])
{
}

void WarpAffineNode::create(std::shared_ptr<Graph> graph)
{
    if(_node)
        return;

    _graph = graph;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")

    _node = vxExtrppNode_WarpAffine(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _outputs[0]->info().width(), _outputs[0]->info().height_batch(),
                                    _x0.default_value(), _y0.default_value(), _o0.default_value(), _x1.default_value(), _y1.default_value(), _o1.default_value());
    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the warp affine (vxExtrppNode_WarpAffine) node failed: "+ TOSTR(status))

    _x0.create(_node);
    _x1.create(_node);
    _y0.create(_node);
    _y1.create(_node);
    _o0.create(_node);
    _o1.create(_node);
}

void WarpAffineNode::init(float x0, float x1, float y0, float y1, float o0, float o1)
{
    _x0.set_param(x0);
    _x1.set_param(x1);
    _y0.set_param(y0);
    _y1.set_param(y1);
    _o0.set_param(o0);
    _o1.set_param(o1);
}

void WarpAffineNode::init(FloatParam* x0, FloatParam* x1, FloatParam* y0, FloatParam* y1, FloatParam* o0, FloatParam* o1)
{
    _x0.set_param(core(x0));
    _x1.set_param(core(x1));
    _y0.set_param(core(y0));
    _y1.set_param(core(y1));
    _o0.set_param(core(o0));
    _o1.set_param(core(o1));
}

void WarpAffineNode::update_parameters()
{
    _x0.update();
    _y0.update();
    _x1.update();
    _y1.update();
    _o0.update();
    _o1.update();
}

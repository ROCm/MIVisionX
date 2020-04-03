#include <vx_ext_rpp.h>
#include <VX/vx_compatibility.h>
#include <graph.h>
#include "node_lens_correction.h"
#include "exception.h"


LensCorrectionNode::LensCorrectionNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs):
        Node(inputs, outputs),
        _strength(STRENGTH_OVX_PARAM_IDX, STRENGTH_RANGE[0], STRENGTH_RANGE[1]),
        _zoom(ZOOM_OVX_PARAM_IDX, ZOOM_RANGE[0], ZOOM_RANGE[1])
{
}

void LensCorrectionNode::create(std::shared_ptr<Graph> graph)
{
    if(_node)
        return;

    _graph = graph;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")

    _node = vxExtrppNode_LensCorrection(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _strength.default_value(), _zoom.default_value());

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the lens correction (vxExtrppNode_LensCorrection) node failed: "+ TOSTR(status))

    _strength.create(_node);
    _zoom.create(_node);
}

void LensCorrectionNode::init(float min, float max)
{
    _strength.set_param(min);
    _zoom.set_param(max);
}

void LensCorrectionNode::init(FloatParam *min, FloatParam *max)
{
    _strength.set_param(core(min));
    _zoom.set_param(core(max));
}

void LensCorrectionNode::update_parameters()
{
    _strength.update();
    _zoom.update();
}



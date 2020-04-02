#include <vx_ext_rpp.h>
#include <VX/vx_compatibility.h>
#include <graph.h>
#include "node_lens_correction.h"
#include "exception.h"


LensCorrectionNode::LensCorrectionNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _strength(STRENGTH_RANGE[0], STRENGTH_RANGE[1]),
        _zoom(ZOOM_RANGE[0], ZOOM_RANGE[1])
{
}

void LensCorrectionNode::create_node()
{
    if(_node)
        return;

    _strength.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _zoom.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _node = vxExtrppNode_LensCorrectionbatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _strength.default_array(), _zoom.default_array(), _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the lens correction (vxExtrppNode_LensCorrection) node failed: "+ TOSTR(status))

}

void LensCorrectionNode::init(float strength, float zoom)
{
    _strength.set_param(strength);
    _zoom.set_param(zoom);
}

void LensCorrectionNode::init(FloatParam* strength, FloatParam* zoom )
{
    _strength.set_param(core(strength));
    _zoom.set_param(core(zoom));
}

void LensCorrectionNode::update_node()
{
    _strength.update_array();
    _zoom.update_array();
}



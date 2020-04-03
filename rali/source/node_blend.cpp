#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_blend.h"
#include "exception.h"


BlendNode::BlendNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _ratio(RATIO_RANGE[0], RATIO_RANGE[1])
{
}
void BlendNode::create_node()
{

    if(_node)
        return;

    if(_inputs.size() < 2)
        THROW("Blend node needs two input images")

    _ratio.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _node = vxExtrppNode_BlendbatchPD(_graph->get(), _inputs[0]->handle(), _inputs[1]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _ratio.default_array(), _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the blend (vxExtrppNode_BlendbatchPD) node failed: "+ TOSTR(status))
}

void BlendNode::init(float ratio)
{
    _ratio.set_param(ratio);
}

void BlendNode::init(FloatParam* ratio)
{
    _ratio.set_param(core(ratio));
}

void BlendNode::update_node()
{
    _ratio.update_array();
}
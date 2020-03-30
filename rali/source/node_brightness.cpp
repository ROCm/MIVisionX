#include <vx_ext_rpp.h>
#include "node_brightness.h"
#include "exception.h"


BrightnessNode::BrightnessNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _alpha(ALPHA_RANGE[0], ALPHA_RANGE[1]),
        _beta (BETA_RANGE[0], BETA_RANGE[1])
{
}

void BrightnessNode::create_node()
{
    if(_node)
        return;

    _alpha.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _beta.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);

    _node = vxExtrppNode_BrightnessbatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _alpha.default_array(), _beta.default_array(), _batch_size);/*A temporary fix for time being*/

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the brightness_batch (vxExtrppNode_BrightnessbatchPD) node failed: "+ TOSTR(status))
}

void BrightnessNode::init( float alpha, int beta)
{
    _alpha.set_param(alpha);
    _beta.set_param(beta);
}

void BrightnessNode::init( FloatParam* alpha, IntParam* beta)
{
    _alpha.set_param(core(alpha));
    _beta.set_param(core(beta));
}


void BrightnessNode::update_node()
{
    _alpha.update_array();
    _beta.update_array();
}


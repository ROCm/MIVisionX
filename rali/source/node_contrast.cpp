#include <vx_ext_rpp.h>
#include "node_contrast.h"
#include "exception.h"

RaliContrastNode::RaliContrastNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _min(CONTRAST_MIN_RANGE[0], CONTRAST_MIN_RANGE[1]),
        _max(CONTRAST_MAX_RANGE[0], CONTRAST_MAX_RANGE[1])
{
}

void RaliContrastNode::create_node()
{

    if(_node)
        return;

    _min.create_array(_graph ,VX_TYPE_UINT32, _batch_size);
    _max.create_array(_graph ,VX_TYPE_UINT32 , _batch_size);

    _node = vxExtrppNode_ContrastbatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _min.default_array(), _max.default_array(), _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the contrast (vxExtrppNode_contrast) node failed: "+ TOSTR(status))
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

void RaliContrastNode::update_node()
{
    _min.update_array();
    _max.update_array();
}


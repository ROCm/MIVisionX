#include <vx_ext_rpp.h>
#include <VX/vx_compatibility.h>
#include <graph.h>
#include "node_hue.h"
#include "exception.h"


HueNode::HueNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _hue(HUE_RANGE[0], HUE_RANGE[1])
{
}

void HueNode::create_node()
{
    if(_node)
        return;

    _hue.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _node = vxExtrppNode_HuebatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _hue.default_array(), _batch_size);
    std::cout<< "Shobana here" << std::endl;
    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the Hue (vxExtrppNode_HueCorrectionbatchPD) node failed: "+ TOSTR(status))

}

void HueNode::init(float hue)
{
    _hue.set_param(hue);
}

void HueNode::init(FloatParam* hue)
{
    _hue.set_param(core(hue));
}

void HueNode::update_node()
{
     _hue.update_array();
}
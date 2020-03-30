#include <vx_ext_rpp.h>
#include <VX/vx_compatibility.h>
#include <graph.h>
#include "node_saturation.h"
#include "exception.h"


SatNode::SatNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _sat(SAT_RANGE[0], SAT_RANGE[1])
{
}

void SatNode::create_node()
{
    if(_node)
        return;
    _sat.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _node = vxExtrppNode_SaturationbatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _sat.default_array(), _batch_size);
    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the Saturation (vxExtrppNodeSaturationbatchPD) node failed: "+ TOSTR(status))

}


void SatNode::init(float sat)
{
    _sat.set_param(sat);
}

void SatNode::init(FloatParam* sat)
{
    _sat.set_param(core(sat));
}

void SatNode::update_node()
{
     _sat.update_array();
}
#include <vx_ext_rpp.h>
#include <VX/vx_compatibility.h>
#include <graph.h>
#include "node_gamma.h"
#include "exception.h"


GammaNode::GammaNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _shift(SHIFT_RANGE[0], SHIFT_RANGE[1])
{
}

void GammaNode::create_node()
{
    if(_node)
        return;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")

    _shift.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _node = vxExtrppNode_GammaCorrectionbatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _shift.default_array(), _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the gamma (vxExtrppNode_GammaCorrectionbatchPD) node failed: "+ TOSTR(status))

}

void GammaNode::init(float shfit)
{
    _shift.set_param(shfit);
}

void GammaNode::init(FloatParam* shfit)
{
    _shift.set_param(core(shfit));
}

void GammaNode::update_node()
{
     _shift.update_array();
}
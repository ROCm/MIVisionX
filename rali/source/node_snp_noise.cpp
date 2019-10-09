#include <vx_ext_rpp.h>
#include <VX/vx_compatibility.h>
#include "node_snp_noise.h"
#include "exception.h"


SnPNoiseNode::SnPNoiseNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs):
        Node(inputs, outputs),
        _sdev(SDEV_OVX_PARAM_IDX, SDEV_RANGE[0], SDEV_RANGE[1])
{
}

void SnPNoiseNode::create(std::shared_ptr<Graph> graph)
{
    if(_node)
        return;

    _graph = graph;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")

    _node = vxExtrppNode_NoiseSnp(_graph->get(), _inputs[0]->handle(), _outputs[0]->handle(), _sdev.default_value());

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the snp noise (vxExtrppNode_NoiseSnp) node failed: "+ TOSTR(status))

    _sdev.create(_node);

}

void SnPNoiseNode::init(float shfit)
{
    _sdev.set_param(shfit);
}

void SnPNoiseNode::init(FloatParam* shfit)
{
    _sdev.set_param(core(shfit));
}

void SnPNoiseNode::update_parameters()
{
    _sdev.update();
}

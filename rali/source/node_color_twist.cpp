#include <vx_ext_rpp.h>
#include "node_color_twist.h"
#include "exception.h"


ColorTwistBatchNode::ColorTwistBatchNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _alpha(ALPHA_RANGE[0], ALPHA_RANGE[1]),
        _beta (BETA_RANGE[0], BETA_RANGE[1]),
        _hue(ALPHA_RANGE[0], HUE_RANGE[1]),
        _sat (BETA_RANGE[0], SAT_RANGE[1])
{
}

void ColorTwistBatchNode::create(std::shared_ptr<Graph> graph)
{
    vx_uint32 *width, *height;
    width = (vx_uint32* ) malloc(sizeof(vx_uint32) * _batch_size);
    height = (vx_uint32* ) malloc(sizeof(vx_uint32) * _batch_size);

    for (uint i = 0; i < _batch_size; i++ ) {
         width[i] = _outputs[0]->info().width();
         height[i] = _outputs[0]->info().height_single();
    }

    vx_status width_status, height_status;
    
    if(_node)
        return;

    _graph = graph;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")

    _alpha.create_array(graph , VX_TYPE_FLOAT32, _batch_size);
    _beta.create_array(graph , VX_TYPE_FLOAT32, _batch_size);
    _hue.create_array(graph , VX_TYPE_FLOAT32, _batch_size);
    _sat.create_array(graph , VX_TYPE_FLOAT32, _batch_size);
   
   
    vx_array width_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    vx_array height_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);

    width_status = vxAddArrayItems(width_array,_batch_size, width, sizeof(vx_uint32));
    height_status = vxAddArrayItems(height_array,_batch_size, height, sizeof(vx_uint32));

    _node = vxExtrppNode_ColorTwistbatchPD(_graph->get(), _inputs[0]->handle(), width_array, height_array, _outputs[0]->handle(), _alpha.default_array(), _beta.default_array(), _hue.default_array(), _sat.default_array(), _batch_size);/*A temporary fix for time being*/

    if(width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the (vxExtrppNode_ColorTwistbatchPD) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the ColorTwist_batch (vxExtrppNode_ColorTwistbatchPD) node failed: "+ TOSTR(status))
}

void ColorTwistBatchNode::init(float alpha, float beta, float hue, float sat)
{
    _alpha.set_param(alpha);
    _beta.set_param(beta);
    _hue.set_param(hue);
    _sat.set_param(sat);
}

void ColorTwistBatchNode::init(FloatParam *alpha, FloatParam *beta, FloatParam *hue, FloatParam *sat)
{
    _alpha.set_param(core(alpha));
    _beta.set_param(core(beta));
    _hue.set_param(core(hue));
    _sat.set_param(core(sat));
}

void ColorTwistBatchNode::update_parameters()
{
    _alpha.update_array();
    _beta.update_array();
    _hue.update_array();
    _sat.update_array();
}


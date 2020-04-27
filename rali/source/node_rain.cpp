#include <vx_ext_rpp.h>
#include <VX/vx_compatibility.h>
#include <graph.h>
#include "node_rain.h"
#include "exception.h"

RainNode::RainNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _rain_value(RAIN_VALUE_RANGE[0], RAIN_VALUE_RANGE[1]),
        _rain_width(RAIN_WIDTH_RANGE[0],RAIN_WIDTH_RANGE[1]),
        _rain_height(RAIN_HEIGHT_RANGE[0],RAIN_HEIGHT_RANGE[0]),
        _rain_transparency(RAIN_TRANSPARENCY_RANGE[0], RAIN_TRANSPARENCY_RANGE[1])
{
}

void RainNode::create_node()
{
    if(_node)
        return;

    _rain_value.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _rain_transparency.create_array(_graph , VX_TYPE_FLOAT32, _batch_size);
    _rain_width.create_array(_graph ,VX_TYPE_UINT32, _batch_size);
    _rain_height.create_array(_graph ,VX_TYPE_UINT32, _batch_size);
    _node = vxExtrppNode_RainbatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _rain_value.default_array(), _rain_width.default_array(),
                                                    _rain_height.default_array(), _rain_transparency.default_array(), _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the rain (vxExtrppNode_Rain) node failed: "+ TOSTR(status))

}

void RainNode::init(float rain_value, int rain_width, int rain_height, float rain_transparency)
{
    _rain_value.set_param(rain_value);
    _rain_width.set_param(rain_width);
    _rain_height.set_param(rain_height);
    _rain_transparency.set_param(rain_transparency);
}

void RainNode::init(FloatParam *rain_value, IntParam *rain_width, IntParam *rain_height, FloatParam *rain_transparency)
{
    _rain_value.set_param(core(rain_value));
    _rain_width.set_param(core(rain_width));
    _rain_height.set_param(core(rain_height));
    _rain_transparency.set_param(core(rain_transparency));
}


void RainNode::update_node()
{
    _rain_height.update_array();
    _rain_width.update_array();
    _rain_value.update_array();
    _rain_transparency.update_array();
}

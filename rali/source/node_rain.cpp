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

void RainNode::create(std::shared_ptr<Graph> graph)
{
    _width.resize(_batch_size);
    _height.resize(_batch_size);

    for (uint i = 0; i < _batch_size; i++ ) {
         _width[i] = _outputs[0]->info().width();
         _height[i] = _outputs[0]->info().height_single();
    }

    vx_status width_status, height_status;

    if(_node)
        return;

    _graph = graph;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")

    _width_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _height_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);

    width_status = vxAddArrayItems(_width_array,_batch_size, _width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(_height_array,_batch_size, _height.data(), sizeof(vx_uint32));

    if(width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the rain (vxExtrppNode_Rain) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))

    _rain_value.create_array(graph , VX_TYPE_FLOAT32, _batch_size);
    _rain_transparency.create_array(graph , VX_TYPE_FLOAT32, _batch_size);
    _rain_width.create_array(graph ,VX_TYPE_UINT32, _batch_size);
    _rain_height.create_array(graph ,VX_TYPE_UINT32, _batch_size);
    _node = vxExtrppNode_RainbatchPD(_graph->get(), _inputs[0]->handle(), _width_array, _height_array, _outputs[0]->handle(), _rain_value.default_array(), _rain_width.default_array(), 
                                                    _rain_height.default_array(), _rain_transparency.default_array(), _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the rain (vxExtrppNode_Rain) node failed: "+ TOSTR(status))

}


void RainNode::update_dimensions()
{
    std::vector<uint> width, height;

    width.resize( _batch_size);
    height.resize( _batch_size);
    for (uint i = 0; i < _batch_size; i++ )
    {
        _width[i] = _inputs[0]->info().get_image_width(i);
        _height[i] = _inputs[0]->info().get_image_height(i);
    }

    vx_status width_status, height_status;
    width_status = vxCopyArrayRange((vx_array)_width_array, 0, _batch_size, sizeof(vx_uint32), _width.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST); //vxAddArrayItems(_width_array,_batch_size, _width, sizeof(vx_uint32));
    height_status = vxCopyArrayRange((vx_array)_height_array, 0, _batch_size, sizeof(vx_uint32), _height.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST); //vxAddArrayItems(_height_array,_batch_size, _height, sizeof(vx_uint32));
    if(width_status != 0 || height_status != 0)
        THROW(" vxCopyArrayRange failed in the rain (vxExtrppNode_Rain) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))
    // TODO: Check the status codes
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


void RainNode::update_parameters()


{
    _rain_height.update_array();
    _rain_width.update_array();
    _rain_value.update_array();
    _rain_transparency.update_array();
}

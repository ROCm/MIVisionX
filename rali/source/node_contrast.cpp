#include <vx_ext_rpp.h>
#include "node_contrast.h"
#include "exception.h"

RaliContrastNode::RaliContrastNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _min(CONTRAST_MIN_RANGE[0], CONTRAST_MIN_RANGE[1]),
        _max(CONTRAST_MAX_RANGE[0], CONTRAST_MAX_RANGE[1])
{
}

void RaliContrastNode::create(std::shared_ptr<Graph> graph)
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
        THROW(" vxAddArrayItems failed in the contrast (vxExtrppNode_contrast) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))

    _min.create_array(graph ,VX_TYPE_UINT32, _batch_size);
    _max.create_array(graph ,VX_TYPE_UINT32 , _batch_size);
    _node = vxExtrppNode_ContrastbatchPD(_graph->get(), _inputs[0]->handle(), _width_array, _height_array, _outputs[0]->handle(), _min.default_array(), _max.default_array(), _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the contrast (vxExtrppNode_contrast) node failed: "+ TOSTR(status))
}

void RaliContrastNode::update_dimensions()
{
    std::vector<uint> width, height;

    width.resize( _batch_size);
    height.resize( _batch_size);
    for (uint i = 0; i < _batch_size; i++)
    {
        _width[i] = _inputs[0]->info().get_image_width(i);
        _height[i] = _inputs[0]->info().get_image_height(i);
    }

    vx_status width_status, height_status;
    width_status = vxCopyArrayRange((vx_array)_width_array, 0, _batch_size, sizeof(vx_uint32), _width.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST); //vxAddArrayItems(_width_array,_batch_size, _width, sizeof(vx_uint32));
    height_status = vxCopyArrayRange((vx_array)_height_array, 0, _batch_size, sizeof(vx_uint32), _height.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST); //vxAddArrayItems(_height_array,_batch_size, _height, sizeof(vx_uint32));
    if(width_status != 0 || height_status != 0)
        THROW(" vxCopyArrayRange failed in the contrast (vxExtrppNode_contrast) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))
    // TODO: Check the status codes
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

void RaliContrastNode::update_parameters()
{
    update_dimensions();
    _min.update_array();
    _max.update_array();
}


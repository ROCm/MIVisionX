#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_flip.h"
#include "exception.h"

FlipNode::FlipNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _flip_axis(FLIP_SIZE[0], FLIP_SIZE[1])
{
}

void FlipNode::create(std::shared_ptr<Graph> graph)
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
        THROW(" vxAddArrayItems failed in the flip (vxExtrppNode_Flip) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))

    _flip_axis.create_array(graph ,VX_TYPE_UINT32 ,_batch_size);
    _node = vxExtrppNode_FlipbatchPD(_graph->get(), _inputs[0]->handle(), _width_array, _height_array, _outputs[0]->handle(), _flip_axis.default_array(), _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the flip (vxExtrppNode_Flip) node failed: "+ TOSTR(status))

}


void FlipNode::update_dimensions()
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
        THROW(" vxCopyArrayRange failed in the flip (vxExtrppNode_Flip) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))
    // TODO: Check the status codes
}

void FlipNode::init(int flip_axis)
{
    _flip_axis.set_param(flip_axis);
}

void FlipNode::init(IntParam* flip_axis)
{
    _flip_axis.set_param(core(flip_axis));
}

void FlipNode::update_parameters()
{
    update_dimensions();
    _flip_axis.update_array();
}

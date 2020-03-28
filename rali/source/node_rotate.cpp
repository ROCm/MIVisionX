#include <vx_ext_rpp.h>
#include "node_rotate.h"
#include "exception.h"


RotateNode::RotateNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _angle(ROTATE_ANGLE_RANGE[0], ROTATE_ANGLE_RANGE[1])
{
}

void RotateNode::create(std::shared_ptr<Graph> graph)
{
    if(_node)
        return;

    _graph = graph;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")
        vx_status width_status, height_status;
    
    _width.resize(_batch_size);
    _height.resize(_batch_size);
    _width_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _height_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    width_status = vxAddArrayItems(_width_array,_batch_size, _width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(_height_array,_batch_size, _height.data(), sizeof(vx_uint32));

    if(width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the rotate (vxExtrppNode_RotatebatchPD) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))
    

    dst_width.resize(_batch_size);
    dst_height.resize(_batch_size);
    for (uint i=0; i < _batch_size; i++ ) {
         dst_width[i] = _outputs[0]->info().width();
         dst_height[i] = _outputs[0]->info().height_single();
    }
    dst_width_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    dst_height_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    width_status = vxAddArrayItems(dst_width_array,_batch_size, dst_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(dst_height_array,_batch_size, dst_height.data(), sizeof(vx_uint32));
    if(width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the rotate (vxExtrppNode_RotatebatchPD) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))

    _angle.create_array(graph , VX_TYPE_FLOAT32, _batch_size);
   _node = vxExtrppNode_RotatebatchPD(_graph->get(), _inputs[0]->handle(),_width_array, _height_array, _outputs[0]->handle(),dst_width_array,dst_height_array, _angle.default_array(), _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the rotate (vxExtrppNode_RotatebatchPD) node failed: "+ TOSTR(status))

}

void RotateNode::update_dimensions()
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
        THROW(" vxCopyArrayRange failed in the rotate (vxExtrppNode_RotatebatchPD) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))
}

void RotateNode::init(float angle)
{
    _angle.set_param(angle);
}

void RotateNode::init(FloatParam* angle)
{
    _angle.set_param(core(angle));
}

void RotateNode::update_parameters()
{
    update_dimensions();
    _angle.update_array();
}

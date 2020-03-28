#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_resize.h"
#include "exception.h"


ResizeNode::ResizeNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs)
{
}

void ResizeNode::create(std::shared_ptr<Graph> graph)
{
    if(_node)
        return;

    _graph = graph;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")

    src_width.resize(_batch_size);
    src_height.resize(_batch_size);
    dst_width.resize(_batch_size);
    dst_height.resize(_batch_size);


    for (uint i=0; i < _batch_size; i++ ) {
         src_width[i] = _inputs[0]->info().width();
         src_height[i] = _inputs[0]->info().height_single();
         dst_width[i] = _outputs[0]->info().width();
         dst_height[i] = _outputs[0]->info().height_single();
    }
    src_width_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    src_height_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    dst_width_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    dst_height_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    vx_status width_status, height_status;
    width_status = vxAddArrayItems(src_width_array,_batch_size, src_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(src_height_array,_batch_size, src_height.data(), sizeof(vx_uint32));
     if(width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the resize (vxExtrppNode_ResizebatchPD) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))


    width_status = vxAddArrayItems(dst_width_array,_batch_size, dst_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(dst_height_array,_batch_size, dst_height.data(), sizeof(vx_uint32));
     if(width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the resize (vxExtrppNode_ResizebatchPD) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))
   _node = vxExtrppNode_ResizebatchPD(_graph->get(), _inputs[0]->handle(),src_width_array,src_height_array, _outputs[0]->handle(),dst_width_array,dst_height_array,_batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the resize (vxExtrppNode_ResizebatchPD) node failed: "+ TOSTR(status))


}


void ResizeNode::update_dimensions()
{
    std::vector<uint> width, height;

    width.resize( _batch_size);
    height.resize( _batch_size);
    for (uint i = 0; i < _batch_size; i++)
    {
        src_width[i] = _inputs[0]->info().get_image_width(i);
        src_height[i] = _inputs[0]->info().get_image_height(i);
    }
    vx_status width_status, height_status;
    width_status = vxCopyArrayRange((vx_array)src_width_array, 0, _batch_size, sizeof(vx_uint32), src_width.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST); //vxAddArrayItems(_width_array,_batch_size, _width, sizeof(vx_uint32));
    height_status = vxCopyArrayRange((vx_array)src_height_array, 0, _batch_size, sizeof(vx_uint32), src_height.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST); //vxAddArrayItems(_height_array,_batch_size, _height, sizeof(vx_uint32));
    if(width_status != 0 || height_status != 0)
        THROW(" vxCopyArrayRange failed in the resize (vxExtrppNode_ResizebatchPD) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))
}

void ResizeNode::update_parameters()
{
    update_dimensions();
}
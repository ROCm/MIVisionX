#include <vx_ext_rpp.h>
#include <VX/vx_compatibility.h>
#include "node_warp_affine.h"
#include "exception.h"


WarpAffineNode::WarpAffineNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _x0(COEFFICIENT_RANGE_1[0], COEFFICIENT_RANGE_1[1]),
        _x1(COEFFICIENT_RANGE_0[0], COEFFICIENT_RANGE_0[1]),
        _y0(COEFFICIENT_RANGE_0[0], COEFFICIENT_RANGE_0[1]),
        _y1(COEFFICIENT_RANGE_1[0], COEFFICIENT_RANGE_1[1]),
        _o0(COEFFICIENT_RANGE_OFFSET[0], COEFFICIENT_RANGE_OFFSET[1]),
        _o1(COEFFICIENT_RANGE_OFFSET[0], COEFFICIENT_RANGE_OFFSET[1])
{
}

void WarpAffineNode::create(std::shared_ptr<Graph> graph)
{
    if(_node)
        return;

    _graph = graph;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")
    vx_status width_status, height_status;
    _affine.resize(6 * _batch_size);    
    _width.resize(_batch_size);
    _height.resize(_batch_size);
    _width_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _height_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    width_status = vxAddArrayItems(_width_array,_batch_size, _width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(_height_array,_batch_size, _height.data(), sizeof(vx_uint32));
    dst_width.resize(_batch_size);
    dst_height.resize(_batch_size);
    uint batch_size = _batch_size;
    for (uint i=0; i < batch_size; i++ ) {
         dst_width[i] = _outputs[0]->info().width();
         dst_height[i] = _outputs[0]->info().height_single();
         _affine[i*6 + 0] = _x0.renew();
         _affine[i*6 + 1] = _y0.renew();
         _affine[i*6 + 2] = _x1.renew();
         _affine[i*6 + 3] = _y1.renew();
         _affine[i*6 + 4] = _o0.renew();
         _affine[i*6 + 5] = _o1.renew();

    }
    dst_width_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    dst_height_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    width_status = vxAddArrayItems(dst_width_array,_batch_size, dst_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(dst_height_array,_batch_size, dst_height.data(), sizeof(vx_uint32));
    if(width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the rotate (vxExtrppNode_WarpAffinePD) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))

    vx_status status;
    _affine_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size * 6);
    status = vxAddArrayItems(_affine_array,_batch_size * 6, _affine.data(), sizeof(vx_float32));
    _node = vxExtrppNode_WarpAffinebatchPD(_graph->get(), _inputs[0]->handle(), _width_array, _height_array, _outputs[0]->handle(),dst_width_array,dst_height_array, 
                                            _affine_array, _batch_size);
    
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Adding the warp affine (vxExtrppNode_WarpAffinePD) node failed: "+ TOSTR(status))
}

void WarpAffineNode::update_dimensions()
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
        THROW(" vxCopyArrayRange failed in the WarpAffine (vxExtrppNode_WarpAffinePD) node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))
}
void WarpAffineNode::update_affine_array()
{
    for (uint i = 0; i < _batch_size; i++ )
    {
        _affine[i*6 + 0] = _x0.renew();
        _affine[i*6 + 1] = _y0.renew();
        _affine[i*6 + 2] = _x1.renew();
        _affine[i*6 + 3] = _y1.renew();
        _affine[i*6 + 4] = _o0.renew();
        _affine[i*6 + 5] = _o1.renew();
    }
    vx_status affine_status;
    affine_status = vxCopyArrayRange((vx_array)_affine_array, 0, _batch_size * 6, sizeof(vx_float32), _affine.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST); //vxAddArrayItems(_width_array,_batch_size, _width, sizeof(vx_uint32));
    if(affine_status != 0)
        THROW(" vxCopyArrayRange failed in the WarpAffine(vxExtrppNode_WarpAffinePD) node: "+ TOSTR(affine_status))
}

void WarpAffineNode::init(float x0, float x1, float y0, float y1, float o0, float o1)
{
    _x0.set_param(x0);
    _x1.set_param(x1);
    _y0.set_param(y0);
    _y1.set_param(y1);
    _o0.set_param(o0);
    _o1.set_param(o1);
}

void WarpAffineNode::init(FloatParam* x0, FloatParam* x1, FloatParam* y0, FloatParam* y1, FloatParam* o0, FloatParam* o1)
{
    _x0.set_param(core(x0));
    _x1.set_param(core(x1));
    _y0.set_param(core(y0));
    _y1.set_param(core(y1));
    _o0.set_param(core(o0));
    _o1.set_param(core(o1));
}

void WarpAffineNode::update_parameters()
{
    update_dimensions();
    update_affine_array();
    // _x0.update();
    // _y0.update();
    // _x1.update();
    // _y1.update();
    // _o0.update();
    // _o1.update();
}

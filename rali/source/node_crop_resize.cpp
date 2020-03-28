#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_crop_resize.h"
#include "exception.h"

CropResizeNode::CropResizeNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _dest_width(_outputs[0]->info().width()),
        _dest_height(_outputs[0]->info().height_batch())
{
    _crop_param = std::make_shared<RandomCropResizeParam>(_batch_size);
}

void CropResizeNode::create(std::shared_ptr<Graph> graph)
{
    if(_node)
        return;

    _graph = graph;

    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")

    if(_dest_width == 0 || _dest_height == 0)
        THROW("Uninitialized destination dimension")
    _crop_param->create_array(graph);
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
        THROW(" vxAddArrayItems failed in the crop resize node (vxExtrppNode_ResizeCropbatchPD    )  node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))
    width_status = vxAddArrayItems(dst_width_array,_batch_size, dst_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(dst_height_array,_batch_size, dst_height.data(), sizeof(vx_uint32));
    if(width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the crop resize node (vxExtrppNode_ResizeCropbatchPD    )  node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))

    _node = vxExtrppNode_ResizeCropbatchPD(_graph->get(), _inputs[0]->handle(), src_width_array, src_height_array, _outputs[0]->handle(), dst_width_array,
                        dst_height_array, _crop_param->x1_arr, _crop_param->y1_arr, _crop_param->x2_arr, _crop_param->y2_arr,_batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Error adding the crop resize node (vxExtrppNode_ResizeCropbatchPD    ) failed: "+TOSTR(status))
}

void CropResizeNode::update_dimensions()
{
    //std::vector<uint> width, height;

    //width.resize( _batch_size);
    //height.resize( _batch_size);
    for (uint i = 0; i < _batch_size; i++)
    {
        src_width[i] = _inputs[0]->info().get_image_width(i);
        src_height[i] = _inputs[0]->info().get_image_height(i);
        _crop_param->set_image_dimensions(i, src_width[i],src_height[i]);
    }
    vx_status width_status, height_status;
    width_status = vxCopyArrayRange((vx_array)src_width_array, 0, _batch_size, sizeof(vx_uint32), src_width.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST); //vxAddArrayItems(_width_array,_batch_size, _width, sizeof(vx_uint32));
    height_status = vxCopyArrayRange((vx_array)src_height_array, 0, _batch_size, sizeof(vx_uint32), src_height.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST); //vxAddArrayItems(_height_array,_batch_size, _height, sizeof(vx_uint32));
    if(width_status != 0 || height_status != 0)
        THROW(" vxCopyArrayRange failed in the crop resize node (vxExtrppNode_ResizeCropbatchPD    )  node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))
}

void CropResizeNode::update_parameters()
{
    update_dimensions();
    _crop_param->update_array();
}

void CropResizeNode::init(float area, float aspect_ratio, float x_center_drift, float y_center_drift)
{
    _crop_param->set_area_coeff(ParameterFactory::instance()->create_single_value_param(area));
    _crop_param->set_aspect_ratio_coeff(ParameterFactory::instance()->create_single_value_param(aspect_ratio));
    _crop_param->set_x_drift(ParameterFactory::instance()->create_single_value_param(x_center_drift));
    _crop_param->set_y_drift(ParameterFactory::instance()->create_single_value_param(y_center_drift));
}


void CropResizeNode::init(FloatParam* area, FloatParam* aspect_ratio, FloatParam *x_center_drift, FloatParam *y_center_drift)
{
    _crop_param->set_area_coeff(core(area));
    _crop_param->set_aspect_ratio_coeff(core(aspect_ratio));
    _crop_param->set_x_drift(core(x_center_drift));
    _crop_param->set_y_drift(core(y_center_drift));
}




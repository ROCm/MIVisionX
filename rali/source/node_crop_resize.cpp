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

void CropResizeNode::create_node()
{
    if(_node)
        return;

    if(_dest_width == 0 || _dest_height == 0)
        THROW("Uninitialized destination dimension")

    _crop_param->create_array(_graph);

    std::vector<uint32_t> dst_roi_width(_batch_size,_outputs[0]->info().width());
    std::vector<uint32_t> dst_roi_height(_batch_size, _outputs[0]->info().height_single());

    _dst_roi_width = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _dst_roi_height = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);

    vx_status width_status, height_status;

    width_status = vxAddArrayItems(_dst_roi_width, _batch_size, dst_roi_width.data(), sizeof(vx_uint32));
    height_status = vxAddArrayItems(_dst_roi_height, _batch_size, dst_roi_height.data(), sizeof(vx_uint32));
    if(width_status != 0 || height_status != 0)
        THROW(" vxAddArrayItems failed in the crop resize node (vxExtrppNode_ResizeCropbatchPD    )  node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))

    _node = vxExtrppNode_ResizeCropbatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _dst_roi_width,
                                           _dst_roi_height, _crop_param->x1_arr, _crop_param->y1_arr, _crop_param->x2_arr, _crop_param->y2_arr, _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Error adding the crop resize node (vxExtrppNode_ResizeCropbatchPD    ) failed: "+TOSTR(status))
}

void CropResizeNode::update_node()
{
    _crop_param->set_image_dimensions(_inputs[0]->info().get_roi_width_vec(), _inputs[0]->info().get_roi_height_vec());
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




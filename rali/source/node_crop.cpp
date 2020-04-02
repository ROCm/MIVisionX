#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_crop.h"
#include "exception.h"

CropNode::CropNode(const std::vector<Image *> &inputs, const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _dest_width(_outputs[0]->info().width()),
        _dest_height(_outputs[0]->info().height_batch())
{
    _crop_param = std::make_shared<CropParam>(_batch_size);
}

void CropNode::create_node()
{
    if(_node)
        return;

    if(_dest_width == 0 || _dest_height == 0)
        THROW("Uninitialized destination dimension")

    _crop_param->create_array(_graph);

    _node = vxExtrppNode_CropPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(), _crop_param->cropw_arr,
                                _crop_param->croph_arr, _crop_param->x1_arr, _crop_param->y1_arr, _batch_size);

    vx_status status;
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Error adding the crop resize node (vxExtrppNode_ResizeCropbatchPD    ) failed: "+TOSTR(status))
}

void CropNode::update_node()
{
    _crop_param->set_image_dimensions(_inputs[0]->info().get_roi_width_vec(), _inputs[0]->info().get_roi_height_vec());
    _crop_param->update_array();
    std::vector<uint32_t> crop_h_dims, crop_w_dims;
    _crop_param->get_crop_dimensions(crop_h_dims, crop_w_dims);
    _outputs[0]->update_image_roi(crop_w_dims, crop_h_dims);
}

void CropNode::init(unsigned int crop_h, unsigned int crop_w, float x_drift, float y_drift)
{
    _crop_param->crop_w = crop_w;
    _crop_param->crop_h = crop_h;
    _crop_param->x1     = 0; 
    _crop_param->y1     = 0;
}

void CropNode::init(unsigned int crop_h, unsigned int crop_w)
{
    _crop_param->crop_w = crop_w;
    _crop_param->crop_h = crop_h;
    _crop_param->x1     = 0; 
    _crop_param->y1     = 0;
    _crop_param->set_centric();
}


void CropNode::init(FloatParam *crop_h_factor, FloatParam  *crop_w_factor, FloatParam *x_drift, FloatParam *y_drift)
{
    _crop_param->set_x_drift_factor(core(x_drift));
    _crop_param->set_y_drift_factor(core(y_drift));
    _crop_param->set_crop_height_factor(core(crop_h_factor));
    _crop_param->set_crop_width_factor(core(crop_w_factor));
    _crop_param->set_random();    
}




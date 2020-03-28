#include <vx_ext_rpp.h>
#include <graph.h>
#include "node_crop_mirror_normalize.h"
#include "exception.h"


CropMirrorNormalizeNode::CropMirrorNormalizeNode(const std::vector<Image *> &inputs,
                                                 const std::vector<Image *> &outputs) :
        Node(inputs, outputs),
        _mirror(MIRROR_RANGE[0], MIRROR_RANGE[1])
{   
}

void CropMirrorNormalizeNode::create(std::shared_ptr<Graph> graph)
{
    if(_node)
        return;
    _graph = graph;
    if(_outputs.empty() || _inputs.empty())
        THROW("Uninitialized input/output arguments")
    if(_crop_h == 0 || _crop_w == 0)
        THROW("Uninitialized destination dimension - Invalid Crop Sizes")
    
    /* Should figure out doing it one more class */
    _src_width.resize(_batch_size);
    _src_height.resize(_batch_size);
    _x1.resize(_batch_size);
    _y1.resize(_batch_size);
    _x2.resize(_batch_size);
    _y2.resize(_batch_size);
    _mean_vx.resize(_batch_size);
    _std_dev_vx.resize(_batch_size);

    for (uint i=0; i < _batch_size; i++ ) {
         _src_width[i] = _inputs[0]->info().width();
         _src_height[i] = _inputs[0]->info().height_single();
         //Assertion to be thrown for out of bound crops
         _x1[i] = 0; // Right Now - Left Top Crops - Mechanism has to be changed for centric crop
         _y1[i] = 0; 
         _x2[i] = _crop_w;
         _y2[i] = _crop_h;
         _mean_vx[i] = _mean;
         _std_dev_vx[i] = _std_dev;
    }
    _src_width_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _src_height_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _x1_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _y1_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _x2_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _y2_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _mean_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size);
    _std_dev_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size);

    vx_status status;
    status  = vxAddArrayItems(_src_width_array,_batch_size, _src_width.data(), sizeof(vx_uint32));
    status |= vxAddArrayItems(_src_height_array,_batch_size, _src_height.data(), sizeof(vx_uint32));
    status |= vxAddArrayItems(_x1_array,_batch_size, _x1.data(), sizeof(vx_uint32));
    status |= vxAddArrayItems(_x2_array,_batch_size, _x2.data(), sizeof(vx_uint32));
    status |= vxAddArrayItems(_y1_array,_batch_size, _y1.data(), sizeof(vx_uint32));
    status |= vxAddArrayItems(_y2_array,_batch_size, _y2.data(), sizeof(vx_uint32));
    status |= vxAddArrayItems(_mean_array,_batch_size, _x2.data(), sizeof(vx_float32));
    status |= vxAddArrayItems(_std_dev_array,_batch_size, _x2.data(), sizeof(vx_float32));
    _mirror.create_array(graph ,VX_TYPE_UINT32, _batch_size);
    if(status != 0)
        THROW(" vxAddArrayItems failed in the crop resize node (vxExtrppNode_CropMirrorNormalizeCropbatchPD    )  node: "+ TOSTR(status) + "  "+ TOSTR(status))
    
    unsigned int chnShift = 0;
    vx_scalar  chnToggle = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&chnShift);
    _node = vxExtrppNode_CropMirrorNormalizebatchPD(_graph->get(), _inputs[0]->handle(), _src_width_array, _src_height_array, _outputs[0]->handle(), 
                                         _x2_array, _y2_array, _x1_array, _y1_array, _mean_array, _std_dev_array, _mirror.default_array() ,chnToggle ,_batch_size);
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Error adding the crop resize node (vxExtrppNode_CropMirrorNormalizeCropbatchPD    ) failed: "+TOSTR(status))
}

void CropMirrorNormalizeNode::update_parameters()
{
    update_dimensions();
    _mirror.update_array();
}

void CropMirrorNormalizeNode::init(int crop_h, int crop_w, float start_x, float start_y, float mean, float std_dev, IntParam *mirror)
{
    _crop_h = crop_h;
    _crop_w = crop_w;
    _mean   = mean;
    _std_dev = std_dev;
    _mirror.set_param(core(mirror));
}

void CropMirrorNormalizeNode::update_dimensions()
{
    for (uint i = 0; i < _batch_size; i++)
    {
        _src_width[i] = _inputs[0]->info().get_image_width(i);
        _src_height[i] = _inputs[0]->info().get_image_height(i);
     }
    vx_status width_status, height_status;
    width_status = vxCopyArrayRange((vx_array)_src_width_array, 0, _batch_size, sizeof(vx_uint32), _src_width.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST); //vxAddArrayItems(_width_array,_batch_size, _width, sizeof(vx_uint32));
    height_status = vxCopyArrayRange((vx_array)_src_height_array, 0, _batch_size, sizeof(vx_uint32), _src_height.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST); //vxAddArrayItems(_height_array,_batch_size, _height, sizeof(vx_uint32));
    if(width_status != 0 || height_status != 0)
        THROW(" vxCopyArrayRange failed in the crop resize node (vxExtrppNode_CropMirrorNormalizeCropbatchPD    )  node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))
}


// CropMirrorNormalizeNode::CropMirrorNormalizeNode(const std::vector<Image*>& inputs, const std::vector<Image*>& outputs, const size_t batch_size):
//         Node(inputs, outputs, batch_size),
//         _dest_width(_outputs[0]->info().width()),
//         _dest_height(_outputs[0]->info().height_batch()),
//         _mean(MEAN_RANGE[0], MEAN_RANGE[1]),
//         _sdev(SDEV_RANGE[0], SDEV_RANGE[1]),
//         _mirror(MIRROR_RANGE[0], MIRROR_RANGE[1])
// {
//     _crop_param = std::make_shared<RandomCropResizeParam>(inputs[0]->info().width() , inputs[0]->info().height_single());
// }

// void CropMirrorNormalizeNode::create(std::shared_ptr<Graph> graph)
// {
//     if(_node)
//         return;

//     _graph = graph;

//     if(_outputs.empty() || _inputs.empty())
//         THROW("Uninitialized input/output arguments")

//     if(_dest_width == 0 || _dest_height == 0)
//         THROW("Uninitialized destination dimension")
//     _crop_param->create_array(graph ,_batch_size);
//     src_width.resize(_batch_size);
//     src_height.resize(_batch_size);
//     for (uint i=0; i < _batch_size; i++ ) {
//          src_width[i] = _inputs[0]->info().width();
//          src_height[i] = _inputs[0]->info().height_single();
//     }
//     src_width_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
//     src_height_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
//     vx_status width_status, height_status;
//     width_status = vxAddArrayItems(src_width_array,_batch_size, src_width.data(), sizeof(vx_uint32));
//     height_status = vxAddArrayItems(src_height_array,_batch_size, src_height.data(), sizeof(vx_uint32));
//     if(width_status != 0 || height_status != 0)
//         THROW(" vxAddArrayItems failed in the crop resize node (vxExtrppNode_CropMirrorNormalizeCropbatchPD    )  node: "+ TOSTR(width_status) + "  "+ TOSTR(height_status))
//     unsigned int chnShift = 0;
//     vx_scalar  chnToggle = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&chnShift);
//     _mean.create_array(graph , VX_TYPE_FLOAT32, _batch_size);
//     _sdev.create_array(graph , VX_TYPE_FLOAT32, _batch_size);
//     _mirror.create_array(graph ,VX_TYPE_UINT32, _batch_size);
//     _node = vxExtrppNode_CropMirrorNormalizebatchPD(_graph->get(), _inputs[0]->handle(), src_width_array, src_height_array, _outputs[0]->handle(), _crop_param->x2_arr,
//                                         _crop_param->y2_arr, _crop_param->x1_arr, _crop_param->y1_arr, _mean.default_array(), _sdev.default_array(), _mirror.default_array() ,chnToggle ,_batch_size);
//     vx_status status;
//     if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
//         THROW("Error adding the crop resize node (vxExtrppNode_CropMirrorNormalizeCropbatchPD    ) failed: "+TOSTR(status))
// }

// void CropMirrorNormalizeNode::init(float area, float aspect_ratio, float x_center_drift, float y_center_drift,float mean, float sdev, int mirror)
// {
//     _crop_param->set_area_coeff(ParameterFactory::instance()->create_single_value_param(area));
//     _crop_param->set_aspect_ratio_coeff(ParameterFactory::instance()->create_single_value_param(area));
//     _crop_param->set_x_drift(ParameterFactory::instance()->create_single_value_param(x_center_drift));
//     _crop_param->set_y_drift(ParameterFactory::instance()->create_single_value_param(y_center_drift));
//     _mean.set_param(mean);
//     _sdev.set_param(sdev);
//     _mirror.set_param(mirror);
// }

// void CropMirrorNormalizeNode::init(FloatParam *area, FloatParam *aspect_ratio,
//                                    FloatParam *x_center_drift, FloatParam *y_center_drift,
//                                    FloatParam* mean, FloatParam* sdev, IntParam* mirror)
// {
//     _crop_param->set_area_coeff(core(area));
//     _crop_param->set_aspect_ratio_coeff(core(aspect_ratio));
//     _crop_param->set_x_drift(core(x_center_drift));
//     _crop_param->set_y_drift(core(y_center_drift));
//     _mean.set_param(core(mean));
//     _sdev.set_param(core(sdev));
//     _mirror.set_param(core(mirror));
// }




// void CropMirrorNormalizeNode::update_parameters()
// {
//     update_dimensions();
//     _crop_param->update_array_for_cmn();
//     _sdev.update_array();
//     _mean.update_array();
//     _mirror.update_array();
// }
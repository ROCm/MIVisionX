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

void CropMirrorNormalizeNode::create_node()
{
    if(_node)
        return;

    if(_crop_h == 0 || _crop_w == 0)
        THROW("Uninitialized destination dimension - Invalid Crop Sizes")
    
    /* Should figure out doing it one more class */

    _x1.resize(_batch_size);
    _y1.resize(_batch_size);
    _x2.resize(_batch_size);
    _y2.resize(_batch_size);
    _mean_vx.resize(_batch_size);
    _std_dev_vx.resize(_batch_size);

    for (uint i=0; i < _batch_size; i++ ) {
         //Assertion to be thrown for out of bound crops
         _x1[i] = 0; // Right Now - Left Top Crops - Mechanism has to be changed for centric crop
         _y1[i] = 0; 
         _x2[i] = _crop_w;
         _y2[i] = _crop_h;
         _mean_vx[i] = _mean;
         _std_dev_vx[i] = _std_dev;
    }

    _x1_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _y1_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _x2_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _y2_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_UINT32, _batch_size);
    _mean_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size);
    _std_dev_array = vxCreateArray(vxGetContext((vx_reference)_graph->get()), VX_TYPE_FLOAT32, _batch_size);

    vx_status status = VX_SUCCESS;

    status |= vxAddArrayItems(_x1_array,_batch_size, _x1.data(), sizeof(vx_uint32));
    status |= vxAddArrayItems(_x2_array,_batch_size, _x2.data(), sizeof(vx_uint32));
    status |= vxAddArrayItems(_y1_array,_batch_size, _y1.data(), sizeof(vx_uint32));
    status |= vxAddArrayItems(_y2_array,_batch_size, _y2.data(), sizeof(vx_uint32));
    status |= vxAddArrayItems(_mean_array,_batch_size, _x2.data(), sizeof(vx_float32));
    status |= vxAddArrayItems(_std_dev_array,_batch_size, _x2.data(), sizeof(vx_float32));
    _mirror.create_array(_graph ,VX_TYPE_UINT32, _batch_size);
    if(status != 0)
        THROW(" vxAddArrayItems failed in the crop resize node (vxExtrppNode_CropMirrorNormalizeCropbatchPD    )  node: "+ TOSTR(status) + "  "+ TOSTR(status))
    
    unsigned int chnShift = 0;
    vx_scalar  chnToggle = vxCreateScalar(vxGetContext((vx_reference)_graph->get()),VX_TYPE_UINT32,&chnShift);
    _node = vxExtrppNode_CropMirrorNormalizebatchPD(_graph->get(), _inputs[0]->handle(), _src_roi_width, _src_roi_height, _outputs[0]->handle(),
                                                    _x2_array, _y2_array, _x1_array, _y1_array, _mean_array, _std_dev_array, _mirror.default_array() , chnToggle , _batch_size);
    if((status = vxGetStatus((vx_reference)_node)) != VX_SUCCESS)
        THROW("Error adding the crop resize node (vxExtrppNode_CropMirrorNormalizeCropbatchPD    ) failed: "+TOSTR(status))
}

void CropMirrorNormalizeNode::update_node()
{
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

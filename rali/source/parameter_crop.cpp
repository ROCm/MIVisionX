#include <cmath>
#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include <graph.h>
#include "parameter_crop.h"
#include "commons.h"

void CropParam::set_x_drift_factor(Parameter<float>* x_drift)
{
    if(!x_drift)
        return ;
    ParameterFactory::instance()->destroy_param(x_drift_factor);
    x_drift_factor = x_drift;
}

void CropParam::set_y_drift_factor(Parameter<float>* y_drift)
{
    if(!y_drift)
        return ;
    ParameterFactory::instance()->destroy_param(y_drift_factor);
    y_drift_factor = y_drift;
}

void CropParam::get_crop_dimensions(std::vector<uint32_t> &crop_w_dim, std::vector<uint32_t> &crop_h_dim)
{   
    crop_h_dim = croph_arr_val;
    crop_w_dim = cropw_arr_val;
}

void CropParam::create_array(std::shared_ptr<Graph> graph)
{
    x1_arr_val.resize(batch_size);
    cropw_arr_val.resize(batch_size);
    y1_arr_val.resize(batch_size);
    croph_arr_val.resize(batch_size);
    x2_arr_val.resize(batch_size);
    y2_arr_val.resize(batch_size);
    in_width.resize(batch_size);
    in_height.resize(batch_size);

    x1_arr =    vxCreateArray(vxGetContext((vx_reference)graph->get()), VX_TYPE_UINT32,batch_size);
    cropw_arr = vxCreateArray(vxGetContext((vx_reference)graph->get()), VX_TYPE_UINT32,batch_size);
    y1_arr =    vxCreateArray(vxGetContext((vx_reference)graph->get()), VX_TYPE_UINT32,batch_size);
    croph_arr = vxCreateArray(vxGetContext((vx_reference)graph->get()), VX_TYPE_UINT32,batch_size);
    x2_arr =    vxCreateArray(vxGetContext((vx_reference)graph->get()), VX_TYPE_UINT32,batch_size);
    y2_arr =    vxCreateArray(vxGetContext((vx_reference)graph->get()), VX_TYPE_UINT32,batch_size);
    vxAddArrayItems(x1_arr,batch_size, x1_arr_val.data(), sizeof(vx_uint32));
    vxAddArrayItems(y1_arr,batch_size, y1_arr_val.data(), sizeof(vx_uint32));
    vxAddArrayItems(cropw_arr,batch_size, cropw_arr_val.data(), sizeof(vx_uint32));
    vxAddArrayItems(croph_arr,batch_size, croph_arr_val.data(), sizeof(vx_uint32));
    vxAddArrayItems(x2_arr,batch_size, x2_arr_val.data(), sizeof(vx_uint32));
    vxAddArrayItems(y2_arr,batch_size, y2_arr_val.data(), sizeof(vx_uint32));
    update_array();
}

void CropParam::update_crop_array()
{
    vx_status status = VX_SUCCESS;
    status = vxCopyArrayRange((vx_array)x1_arr, 0, batch_size, sizeof(vx_uint32), x1_arr_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != VX_SUCCESS)
        WRN("ERROR: vxCopyArrayRange x1_arr failed " +TOSTR(status));
    status = vxCopyArrayRange((vx_array)y1_arr, 0, batch_size, sizeof(vx_uint32), y1_arr_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != VX_SUCCESS)
        WRN("ERROR: vxCopyArrayRange x1_arr failed " +TOSTR(status));
    status = vxCopyArrayRange((vx_array)cropw_arr, 0, batch_size, sizeof(vx_uint32), cropw_arr_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != VX_SUCCESS)
        WRN("ERROR: vxCopyArrayRange x1_arr failed " +TOSTR(status));
    status = vxCopyArrayRange((vx_array)croph_arr, 0, batch_size, sizeof(vx_uint32), croph_arr_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != VX_SUCCESS)
        WRN("ERROR: vxCopyArrayRange x1_arr failed " +TOSTR(status));
    status = vxCopyArrayRange((vx_array)x2_arr, 0, batch_size, sizeof(vx_uint32), x2_arr_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != VX_SUCCESS)
        WRN("ERROR: vxCopyArrayRange x1_arr failed " +TOSTR(status));
    status = vxCopyArrayRange((vx_array)y2_arr, 0, batch_size, sizeof(vx_uint32), y2_arr_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != VX_SUCCESS)
        WRN("ERROR: vxCopyArrayRange x1_arr failed " +TOSTR(status));
}

Parameter<float> *CropParam::default_x_drift_factor()
{
    return ParameterFactory::instance()->create_uniform_float_rand_param(CROP_X_DRIFT_RANGE[0],
                                                                         CROP_X_DRIFT_RANGE[1])->core;
}

Parameter<float> *CropParam::default_y_drift_factor()
{
    return ParameterFactory::instance()->create_uniform_float_rand_param(CROP_Y_DRIFT_RANGE[0],                                                                        
                                                                    CROP_Y_DRIFT_RANGE[1])->core;
}

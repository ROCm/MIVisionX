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

void CropParam::set_crop_height_factor(Parameter<float>* crop_h_factor)
{
    if(!crop_h_factor)
        return ;
    ParameterFactory::instance()->destroy_param(crop_height_factor);
    crop_height_factor = crop_h_factor;
}

void CropParam::set_crop_width_factor(Parameter<float>* crop_w_factor)
{
    if(!crop_w_factor)
        return ;
    ParameterFactory::instance()->destroy_param(crop_width_factor);
    crop_width_factor = crop_w_factor;
}

void CropParam::get_crop_dimensions(std::vector<uint32_t> &crop_w_dim, std::vector<uint32_t> &crop_h_dim)
{   
    crop_h_dim = cropw_arr_val;
    crop_w_dim = cropw_arr_val;
}


void CropParam::create_array(std::shared_ptr<Graph> graph)
{
    x1_arr_val.resize(batch_size);
    cropw_arr_val.resize(batch_size);
    y1_arr_val.resize(batch_size);
    croph_arr_val.resize(batch_size);
    in_width.resize(batch_size);
    in_height.resize(batch_size);

    x1_arr =    vxCreateArray(vxGetContext((vx_reference)graph->get()), VX_TYPE_UINT32,batch_size);
    cropw_arr = vxCreateArray(vxGetContext((vx_reference)graph->get()), VX_TYPE_UINT32,batch_size);
    y1_arr =    vxCreateArray(vxGetContext((vx_reference)graph->get()), VX_TYPE_UINT32,batch_size);
    croph_arr = vxCreateArray(vxGetContext((vx_reference)graph->get()), VX_TYPE_UINT32,batch_size);

    vxAddArrayItems(x1_arr,batch_size, x1_arr_val.data(), sizeof(vx_uint32));
    vxAddArrayItems(y1_arr,batch_size, y1_arr_val.data(), sizeof(vx_uint32));
    vxAddArrayItems(cropw_arr,batch_size, cropw_arr_val.data(), sizeof(vx_uint32));
    vxAddArrayItems(croph_arr,batch_size, croph_arr_val.data(), sizeof(vx_uint32));
    update_array();
}

void CropParam::update_array()
{
    vx_status status = VX_SUCCESS;
    fill_values();
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
}

void CropParam::fill_values()
{
    int cropw_temp = 0, croph_temp;

    if(!(_random))
        {
        for (uint img_idx = 0; img_idx < batch_size; img_idx++)
            {
                if(crop_w >= in_width[img_idx])    cropw_temp = in_width[img_idx] - 1;
		else 	                           cropw_temp = crop_w;

                if(crop_h >= in_height[img_idx])   croph_temp = in_height[img_idx] - 1;
		else                               croph_temp = crop_h;

                if(_centric)
		{
                    x1_arr_val[img_idx] =  in_width[img_idx] /2 - (cropw_temp / 2); 
                    y1_arr_val[img_idx] =  in_height[img_idx] / 2 - (croph_temp / 2);
                    cropw_arr_val[img_idx] = cropw_temp;
                    croph_arr_val[img_idx] = croph_temp;
                }
                else
                {
                    x1_arr_val[img_idx] =  0; 
                    y1_arr_val[img_idx] =  0;
                    cropw_arr_val[img_idx] = cropw_temp;
                    croph_arr_val[img_idx] = croph_temp;
                }      
            }
        }
        else
        {
            for (uint img_idx = 0; img_idx < batch_size; img_idx++)
                {
                    // Left-Top Random Crop
                    crop_height_factor->renew();
                    float crop_h_factor_ = crop_height_factor->get();
                    crop_width_factor->renew();
                    float crop_w_factor_ = crop_width_factor->get();
    
                    x1_arr_val[img_idx] =  0 ; 
                    y1_arr_val[img_idx] =  0;
                    cropw_arr_val[img_idx] = static_cast<size_t> (crop_w_factor_ * in_width[img_idx]);
                    croph_arr_val[img_idx] = static_cast<size_t> (crop_h_factor_ * in_height[img_idx]);
		    if(cropw_arr_val[img_idx] >= in_width[img_idx])  cropw_arr_val[img_idx] = in_width[img_idx] - 1;
                    if(croph_arr_val[img_idx] >= in_height[img_idx]) croph_arr_val[img_idx] = in_height[img_idx] - 1;

                }
        }    
}

Parameter<float> *CropParam::default_x_drift_factor()
{

    return ParameterFactory::instance()->create_uniform_float_rand_param(CROP_X_DRIFT_RANGE[0],
                                                                         CROP_X_DRIFT_RANGE[1])->core;
}

Parameter<float> *CropParam::default_y_drift_factor()
{
    return ParameterFactory::instance()->create_uniform_float_rand_param(CROP_Y_DRIFT_RANGE[0],                                                                         CROP_Y_DRIFT_RANGE[1])->core;
}

Parameter<float> *CropParam::default_crop_height_factor()
{
    return ParameterFactory::instance()->create_uniform_float_rand_param(CROP_HEIGHT_FACTOR_RANGE[0],
                                                                         CROP_HEIGHT_FACTOR_RANGE[1])->core;
}

Parameter<float> *CropParam::default_crop_width_factor()
{
    return ParameterFactory::instance()->create_uniform_float_rand_param(CROP_WIDTH_FACTOR_RANGE[0],
                                                                         CROP_WIDTH_FACTOR_RANGE[1])->core;
}

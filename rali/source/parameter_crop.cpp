#include <cmath>
#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include <graph.h>
#include "parameter_crop.h"
#include "commons.h"

void CropParam::set_x_drift(Parameter<float>* x_drift)
{
    if(!x_drift)
        return;

    ParameterFactory::instance()->destroy_param(x_center_drift);
    x_center_drift = x_drift;
}

void CropParam::set_y_drift(Parameter<float>* y_drift)
{
    if(!y_drift)
        return;

    ParameterFactory::instance()->destroy_param(y_center_drift);
    y_center_drift = y_drift;
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


void CropParam::create_array(std::shared_ptr<Graph> graph)
{
    x1_arr_val.resize(batch_size);
    cropw_arr_val.resize(batch_size);
    y1_arr_val.resize(batch_size);
    croph_arr_val.resize(batch_size);
    in_width.resize(batch_size);
    in_height.resize(batch_size);

    x1_arr =    vxCreateArray(vxGetContext((vx_reference)graph->get()), VX_TYPE_UINT64,batch_size);
    cropw_arr = vxCreateArray(vxGetContext((vx_reference)graph->get()), VX_TYPE_UINT64,batch_size);
    y1_arr =    vxCreateArray(vxGetContext((vx_reference)graph->get()), VX_TYPE_UINT64,batch_size);
    croph_arr = vxCreateArray(vxGetContext((vx_reference)graph->get()), VX_TYPE_UINT64,batch_size);

    vxAddArrayItems(x1_arr,batch_size, x1_arr_val.data(), sizeof(size_t));
    vxAddArrayItems(y1_arr,batch_size, y1_arr_val.data(), sizeof(size_t));
    vxAddArrayItems(cropw_arr,batch_size, cropw_arr_val.data(), sizeof(size_t));
    vxAddArrayItems(croph_arr,batch_size, croph_arr_val.data(), sizeof(size_t));
    update_array();
}

void CropParam::update_array()
{
    vx_status status = VX_SUCCESS;
    if(centric)
    {
        for (uint img_idx = 0; img_idx < batch_size; img_idx++)
        {
            //x_center_drift->renew();
            //float x_center = x_center_drift->get();
            //y_center_drift->renew();
            //y_center_drift->renew();
            //float y_center = y_center_drift->get();
            //fill_values(img_idx); - following lines can be moved in fuction
            x1_arr_val[img_idx] =  in_width[img_idx] /2 - (crop_w / 2); // If it  is a normalized coordinatess, then make it float * width etc.
            y1_arr_val[img_idx] =  in_height[img_idx] / 2 - (crop_h / 2);
            cropw_arr_val[img_idx] = crop_w;
            croph_arr_val[img_idx] = crop_h;
            /* Check for Out of bounds condition  and take care of the situation */
        }
    }
    else
    {
        for (uint img_idx = 0; img_idx < batch_size; img_idx++)
        {
            //x_center_drift->renew();
            //float x_center = x_center_drift->get();
            //y_center_drift->renew();
            //y_center_drift->renew();
            //float y_center = y_center_drift->get();
            //fill_values(img_idx); - following lines can be moved in fuction
            x1_arr_val[img_idx] =  x1 ; // If it  is a normalized coordinatess, then make it float * width etc.
            y1_arr_val[img_idx] =  y1;
            cropw_arr_val[img_idx] = crop_w;
            croph_arr_val[img_idx] = crop_h;
        }
    }
    
    

    status = vxCopyArrayRange((vx_array)x1_arr, 0, batch_size, sizeof(size_t), x1_arr_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != VX_SUCCESS)
        WRN("ERROR: vxCopyArrayRange x1_arr failed " +TOSTR(status));
    status = vxCopyArrayRange((vx_array)y1_arr, 0, batch_size, sizeof(size_t), y1_arr_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != VX_SUCCESS)
        WRN("ERROR: vxCopyArrayRange x1_arr failed " +TOSTR(status));
    status = vxCopyArrayRange((vx_array)cropw_arr, 0, batch_size, sizeof(size_t), cropw_arr_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != VX_SUCCESS)
        WRN("ERROR: vxCopyArrayRange x1_arr failed " +TOSTR(status));
    status = vxCopyArrayRange((vx_array)croph_arr, 0, batch_size, sizeof(size_t), croph_arr_val.data(), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);
    if(status != VX_SUCCESS)
        WRN("ERROR: vxCopyArrayRange x1_arr failed " +TOSTR(status));
}

// void CropParam::fill_values(unsigned int img_idx)
// {
//    x1[img_idx] = 0; // x_start_drift * _width
//    y1[img_idx] = 0; // y_start_drift * _height
//    // crop_h[img_idx] =
//    // crop_w[img_idx] = 

//   // in_height[img_idx] = crop_h[img_idx];
//    //in_width[img_idx] = crop_w[img_idx];
// }


Parameter<float> *CropParam::default_x_drift()
{
    return ParameterFactory::instance()->create_uniform_float_rand_param(CROP_X_DRIFT_RANGE[0],
                                                                         CROP_X_DRIFT_RANGE[1])->core;
}

Parameter<float> *CropParam::default_y_drift()
{
    return ParameterFactory::instance()->create_uniform_float_rand_param(CROP_Y_DRIFT_RANGE[0],
                                                                         CROP_Y_DRIFT_RANGE[1])->core;
}

Parameter<float> *CropParam::default_crop_height_factor()
{
    return ParameterFactory::instance()->create_uniform_float_rand_param(CROP_HEIGHT_FACTOR_RANGE[0],
                                                                         CROP_WIDTH_FACTOR_RANGE[1])->core;
}

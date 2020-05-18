#pragma once
#include "parameter_crop.h"

class RaliCropParam : public CropParam
{
public:
    RaliCropParam() = delete;
    RaliCropParam(unsigned int batch_size): CropParam(batch_size)
    {
        crop_height_factor = default_crop_height_factor();
        crop_width_factor  = default_crop_width_factor();
    }
    unsigned int  crop_w, crop_h, crop_d;
    void set_crop_height_factor(Parameter<float>* crop_h_factor);
    void set_crop_width_factor(Parameter<float>*  crop_w_factor);
    void update_array() override;
private:
    constexpr static float CROP_HEIGHT_FACTOR_RANGE[2]  = {0.55, 0.95}; 
    constexpr static float CROP_WIDTH_FACTOR_RANGE[2]   = {0.55, 0.95};
    Parameter<float>* default_crop_height_factor();
    Parameter<float>* default_crop_width_factor();
    Parameter<float> *crop_height_factor, *crop_width_factor;
    void fill_crop_dims() override;
};

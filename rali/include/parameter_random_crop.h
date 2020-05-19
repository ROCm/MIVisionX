#pragma once
#include "parameter_crop.h"

class RaliRandomCropParam : public CropParam
{
public:
    RaliRandomCropParam() = delete;
    RaliRandomCropParam(unsigned int batch_size): CropParam(batch_size)
    {
        area_factor   = default_area_factor();
        aspect_ratio  = default_aspect_ratio();
    }
    void set_area_factor(Parameter<float>*   crop_h_factor);
    void set_aspect_ratio(Parameter<float>*  crop_w_factor);
    void update_array() override;
private:
    constexpr static float AREA_FACTOR_RANGE[2]  = {0.08, 0.99}; 
    constexpr static float ASPECT_RATIO_RANGE[2] = {0.7500, 1.333};
    Parameter<float>* default_area_factor();
    Parameter<float>* default_aspect_ratio();
    Parameter<float> *area_factor, *aspect_ratio;
    void fill_crop_dims() override;
};


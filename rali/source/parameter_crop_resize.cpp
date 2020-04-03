
#include <cmath>
#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include "parameter_crop_resize.h"
#include "commons.h"
void RandomCropResizeParam::set_area_coeff(Parameter<float>* area)
{
    if(!area)
        return;

    ParameterFactory::instance()->destroy_param(area_coeff);
    area_coeff = area;
}
void RandomCropResizeParam::set_x_drift(Parameter<float>* x_drift)
{
    if(!x_drift)
        return;

    ParameterFactory::instance()->destroy_param(x_center_drift);
    x_center_drift = x_drift;
}
void RandomCropResizeParam::set_y_drift(Parameter<float>* y_drift)
{
    if(!y_drift)
        return;

    ParameterFactory::instance()->destroy_param(y_center_drift);
    y_center_drift = y_drift;
}


void RandomCropResizeParam::update()
{
    calculate_area(
            area_coeff->get(),
            x_center_drift->get(),
            y_center_drift->get());

    vx_status status = VX_SUCCESS;

    if ((status = vxWriteScalarValue(x1_vx_scalar, &x1)) != VX_SUCCESS)
        WRN("ERROR: vxWriteScalarValue x1 failed " +TOSTR(status));

    if ((status = vxWriteScalarValue(x2_vx_scalar, &x2)) != VX_SUCCESS)
        WRN("ERROR: vxWriteScalarValue x1 failed " +TOSTR(status));

    if ((status = vxWriteScalarValue(y1_vx_scalar, &y1)) != VX_SUCCESS)
        WRN("ERROR: vxWriteScalarValue x1 failed " +TOSTR(status));

    if ((status = vxWriteScalarValue(y2_vx_scalar, &y2)) != VX_SUCCESS)
        WRN("ERROR: vxWriteScalarValue x1 failed " +TOSTR(status));
}

void RandomCropResizeParam::calculate_area(float area_coeff_, float x_center_drift_, float y_center_drift_)
{

// +-----------------------------------------> X direction
// |  ___________________________________
// |  |   (x1,y1)      |                |
// |  |    +-----------|-----------+    |
// |  |    |           |           |    |
// |  -----------------o-----------------
// |  |    |           |           |    |
// |  |    +-----------|-----------+    |
// |  |                |        (x2,y2) |
// |  +++++++++++++++++++++++++++++++++++
// |
// V Y directoin

    auto bound = [](float arg, float min , float max)
    {
        if( arg < min)
            return min;
        if( arg > max)
            return max;
        return arg;
    };

    auto y_center = in_height / 2;
    auto x_center = in_width / 2;


    float length_coeff = std::sqrt(bound(area_coeff_, RandomCropResizeParam::MIN_RANDOM_AREA_COEFF, 1.0));

    auto cropped_width = (size_t)(length_coeff  * (float)in_width);
    auto cropped_height= (size_t)(length_coeff * (float)in_height);

    size_t y_max_drift = (in_height - cropped_height) / 2;
    size_t x_max_drift = (in_width - cropped_width ) / 2;


    size_t no_drift_y1 = y_center - cropped_height/2;
    size_t no_drift_x1 = x_center - cropped_width/2;


    float x_drift_coeff = bound(x_center_drift_, -1.0, 1.0);// in [-1 1] range
    float y_drift_coeff = bound(y_center_drift_, -1.0, 1.0);// in [-1 1] range


    x1 = (size_t)((float)no_drift_x1 + x_drift_coeff * (float)x_max_drift);
    y1 = (size_t)((float)no_drift_y1 + y_drift_coeff * (float)y_max_drift);
    x2 = x1 + cropped_width;
    y2 = y1 + cropped_height;

    auto check_bound = [](int arg, int min , int max)
    {
        return arg < min || arg > max;
    };

    if(check_bound(x1, 0, in_width) || check_bound(x2, 0, in_width) || check_bound(y1, 0, in_height) || check_bound(y2, 0, in_height))
        // TODO: proper action required here
        WRN("Wrong crop area calculation")
}

void RandomCropResizeParam::create_scalars(vx_node node)
{
    auto ref = vxGetParameterByIndex(node, RESIZE_CROP_X1_OVX_PARAM_IDX);
    if(vxQueryParameter(ref, VX_PARAMETER_ATTRIBUTE_REF, &x1_vx_scalar, sizeof(vx_scalar)) != VX_SUCCESS)
        THROW("Extracting RESIZE_CROP_X1_OVX_PARAM_IDX failed")

    ref = vxGetParameterByIndex(node, RESIZE_CROP_Y1_OVX_PARAM_IDX);
    if(vxQueryParameter(ref, VX_PARAMETER_ATTRIBUTE_REF, &y1_vx_scalar, sizeof(vx_scalar)) != VX_SUCCESS)
        THROW("Extracting RESIZE_CROP_Y1_OVX_PARAM_IDX failed")

    ref = vxGetParameterByIndex(node, RESIZE_CROP_X2_OVX_PARAM_IDX);
    if(vxQueryParameter(ref, VX_PARAMETER_ATTRIBUTE_REF, &x2_vx_scalar, sizeof(vx_scalar)) != VX_SUCCESS)
        THROW("Extracting RESIZE_CROP_X2_OVX_PARAM_IDX failed")

    ref = vxGetParameterByIndex(node, RESIZE_CROP_Y2_OVX_PARAM_IDX);
    if(vxQueryParameter(ref, VX_PARAMETER_ATTRIBUTE_REF, &y2_vx_scalar, sizeof(vx_scalar)) != VX_SUCCESS)
        THROW("Extracting RESIZE_CROP_Y2_OVX_PARAM_IDX failed")


}

Parameter<float> *RandomCropResizeParam::default_area()
{
    return ParameterFactory::instance()->create_uniform_float_rand_param(CROP_AREA_RANGE[0],
                                                                         CROP_AREA_RANGE[1])->core;
}

Parameter<float> *RandomCropResizeParam::default_x_drift()
{
    return ParameterFactory::instance()->create_uniform_float_rand_param(CROP_AREA_X_DRIFT_RANGE[0],
                                                                         CROP_AREA_X_DRIFT_RANGE[1])->core;
}

Parameter<float> *RandomCropResizeParam::default_y_drift()
{
    return ParameterFactory::instance()->create_uniform_float_rand_param(CROP_AREA_Y_DRIFT_RANGE[0],
                                                                         CROP_AREA_Y_DRIFT_RANGE[1])->core;
}
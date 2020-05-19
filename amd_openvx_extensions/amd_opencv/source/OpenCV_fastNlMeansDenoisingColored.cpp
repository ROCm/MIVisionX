/*
Copyright (c) 2015 - 2020 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include"internal_publishKernels.h"

/************************************************************************************************************
input parameter validator.
param [in] node The handle to the node.
param [in] index The index of the parameter to validate.
*************************************************************************************************************/
static vx_status VX_CALLBACK CV_fastNlMeansDenoisingColored_InputValidator(vx_node node, vx_uint32 index)
{
    vx_status status = VX_SUCCESS;
    vx_parameter param = vxGetParameterByIndex(node, index);

    if (index == 0)
    {
        vx_image image;
        vx_df_image df_image = VX_DF_IMAGE_VIRT;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &image, sizeof(vx_image)));
        STATUS_ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
        if (df_image != VX_DF_IMAGE_RGB)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseImage(&image);
    }

    else if (index == 1)
    {
        vx_image image;
        vx_df_image df_image = VX_DF_IMAGE_VIRT;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &image, sizeof(vx_image)));
        STATUS_ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
        if (df_image != VX_DF_IMAGE_RGB)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseImage(&image);
    }

    else if (index == 2)
    {
        vx_scalar scalar = 0;
        vx_enum type = 0;
        vx_float32 value = 0;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
        STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
        STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
        if (value < 0 || type != VX_TYPE_FLOAT32)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseScalar(&scalar);
    }

    else if (index == 3)
    {
        vx_scalar scalar = 0;
        vx_enum type = 0;
        vx_float32 value = 0;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
        STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
        STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
        if (value < 0 || type != VX_TYPE_FLOAT32)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseScalar(&scalar);
    }

    else if (index == 4)
    {
        vx_scalar scalar = 0;
        vx_enum type = 0;
        vx_int32 value = 0;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
        STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
        STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
        if (value < 0 || type != VX_TYPE_INT32)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseScalar(&scalar);
    }

    else if (index == 5)
    {
        vx_scalar scalar = 0;
        vx_enum type = 0;
        vx_int32 value = 0;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
        STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
        STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
        if (value < 0 || type != VX_TYPE_INT32)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseScalar(&scalar);
    }

    vxReleaseParameter(&param);
    return status;
}

/************************************************************************************************************
output parameter validator.
*************************************************************************************************************/
static vx_status VX_CALLBACK CV_fastNlMeansDenoisingColored_OutputValidator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
    vx_status status = VX_SUCCESS;
    if (index == 1)
    {
        vx_parameter output_param = vxGetParameterByIndex(node, 1);
        vx_image output;
        vx_uint32 width = 0, height = 0;
        vx_df_image format = VX_DF_IMAGE_VIRT;

        STATUS_ERROR_CHECK(vxQueryParameter(output_param, VX_PARAMETER_ATTRIBUTE_REF, &output, sizeof(vx_image)));
        STATUS_ERROR_CHECK(vxQueryImage(output, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
        STATUS_ERROR_CHECK(vxQueryImage(output, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
        STATUS_ERROR_CHECK(vxQueryImage(output, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));

        if (format != VX_DF_IMAGE_RGB)
        {
            status = VX_ERROR_INVALID_VALUE;
        }

        STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
        STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
        STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));

        vxReleaseImage(&output);
        vxReleaseParameter(&output_param);
    }
    return status;
}

/************************************************************************************************************
Execution Kernel
*************************************************************************************************************/
static vx_status VX_CALLBACK CV_fastNlMeansDenoisingColored_Kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    vx_status status = VX_SUCCESS;

    vx_image image_in = (vx_image) parameters[0];
    vx_image image_out = (vx_image) parameters[1];
    vx_scalar H = (vx_scalar) parameters[2];
    vx_scalar H_COLOR = (vx_scalar) parameters[3];
    vx_scalar Template_WS = (vx_scalar) parameters[4];
    vx_scalar Search_WS = (vx_scalar) parameters[5];
    Mat *mat, bl;
    int search_ws, template_ws;
    float h, h_color;
    vx_float32 value_f = 0;
    vx_int32 value = 0;

    //Extracting Values from the Scalar into Ksize and Ddepth
    STATUS_ERROR_CHECK(vxReadScalarValue(H, &value_f));
    h = value_f;
    STATUS_ERROR_CHECK(vxReadScalarValue(H_COLOR, &value_f));
    h_color = value_f;
    STATUS_ERROR_CHECK(vxReadScalarValue(Template_WS, &value));
    template_ws = value;
    STATUS_ERROR_CHECK(vxReadScalarValue(Search_WS, &value));
    search_ws = value;

    //Converting VX Image to OpenCV Mat
    STATUS_ERROR_CHECK(match_vx_image_parameters(image_in, image_out));
    STATUS_ERROR_CHECK(VX_to_CV_Image(&mat, image_in));

    //Compute using OpenCV
    cv::fastNlMeansDenoisingColored(*mat, bl, h, h_color, template_ws, search_ws);

    //Converting OpenCV Mat into VX Image
    STATUS_ERROR_CHECK(CV_to_VX_Image(image_out, &bl));

    return status;
}

/************************************************************************************************************
Function to Register the Kernel for Publish
*************************************************************************************************************/
vx_status CV_fastNlMeansDenoisingColored_Register(vx_context context)
{
    vx_status status = VX_SUCCESS;
    vx_kernel Kernel = vxAddKernel(context,
                                   "org.opencv.fastnlmeansdenoisingcolored",
                                   VX_KERNEL_OPENCV_FAST_NL_MEANS_DENOISING_COLORED,
                                   CV_fastNlMeansDenoisingColored_Kernel,
                                   6,
                                   CV_fastNlMeansDenoisingColored_InputValidator,
                                   CV_fastNlMeansDenoisingColored_OutputValidator,
                                   nullptr,
                                   nullptr);

    if (Kernel)
    {
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 1, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(Kernel));
    }

    if (status != VX_SUCCESS)
    {
exit:
        vxRemoveKernel(Kernel);
        return VX_FAILURE;
    }

    return status;
}

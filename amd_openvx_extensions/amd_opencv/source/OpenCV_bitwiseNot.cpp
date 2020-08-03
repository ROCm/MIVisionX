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
vx_status VX_CALLBACK CV_bitwise_not_InputValidator(vx_node node, vx_uint32 index)
{
    vx_status status = VX_SUCCESS;
    vx_parameter param = vxGetParameterByIndex(node, index);

    if (index == 0)
    {
        vx_image image;
        vx_df_image df_image = VX_DF_IMAGE_VIRT;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &image, sizeof(vx_image)));
        STATUS_ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
        if (df_image != VX_DF_IMAGE_U8 && df_image != VX_DF_IMAGE_S16)
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
        if (df_image != VX_DF_IMAGE_U8 && df_image != VX_DF_IMAGE_S16)
        {
            status = VX_ERROR_INVALID_VALUE;
        }
        vxReleaseImage(&image);
    }

    vxReleaseParameter(&param);
    return status;
}

/************************************************************************************************************
output parameter validator.
*************************************************************************************************************/
vx_status VX_CALLBACK CV_bitwise_not_OutputValidator(vx_node node, vx_uint32 index, vx_meta_format meta)
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

        if (format != VX_DF_IMAGE_U8 && format != VX_DF_IMAGE_S16)
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
vx_status VX_CALLBACK CV_bitwise_not_Kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    vx_status status = VX_SUCCESS;
    vx_image image_in = (vx_image) parameters[0];
    vx_image image_out = (vx_image) parameters[1];

    Mat *mat, bl;

    //Converting VX Image to OpenCV Mat 1
    STATUS_ERROR_CHECK(match_vx_image_parameters(image_in, image_out));
    STATUS_ERROR_CHECK(VX_to_CV_Image(&mat, image_in));

    //Compute using OpenCV
    cv::bitwise_not(*mat, bl);

    //Converting OpenCV Mat into VX Image
    STATUS_ERROR_CHECK(CV_to_VX_Image(image_out, &bl));

    return status;
}

/************************************************************************************************************
Function to Register the Kernel for Publish
*************************************************************************************************************/
vx_status CV_bitwise_not_Register(vx_context context)
{
    vx_status status = VX_SUCCESS;
    vx_kernel kernel = vxAddKernel(context,
                                   "org.opencv.bitwise_not",
                                   VX_KERNEL_OPENCV_BITWISE_NOT,
                                   CV_bitwise_not_Kernel,
                                   2,
                                   CV_bitwise_not_InputValidator,
                                   CV_bitwise_not_OutputValidator,
                                   nullptr,
                                   nullptr);

    if (kernel)
    {
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
        PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
    }

    if (status != VX_SUCCESS)
    {
exit:
        vxRemoveKernel(kernel);
        return VX_FAILURE;
    }

    return status;
}

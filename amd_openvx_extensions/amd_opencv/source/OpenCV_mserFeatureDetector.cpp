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

/*!***********************************************************************************************************
input parameter validator.
param [in] node The handle to the node.
param [in] index The index of the parameter to validate.
*************************************************************************************************************/
static vx_status VX_CALLBACK CV_MSER_feature_detector_InputValidator(vx_node node, vx_uint32 index)
{
	vx_status status = VX_SUCCESS;
	vx_parameter param = vxGetParameterByIndex(node, index);

	if (index == 0)
	{
		vx_image image;
		vx_df_image df_image = VX_DF_IMAGE_VIRT;
		STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &image, sizeof(vx_image)));
		STATUS_ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
		if (df_image != VX_DF_IMAGE_U8)
			status = VX_ERROR_INVALID_VALUE;
		vxReleaseImage(&image);
	}

	else if (index == 1)
	{
		vx_array array; vx_size size = 0;
		STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &array, sizeof(array)));
		STATUS_ERROR_CHECK(vxQueryArray(array, VX_ARRAY_ATTRIBUTE_CAPACITY, &size, sizeof(size)));
		vxReleaseArray(&array);
	}

	if (index == 2)
	{
		vx_image image;
		vx_df_image df_image = VX_DF_IMAGE_VIRT;
		STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &image, sizeof(vx_image)));
		STATUS_ERROR_CHECK(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)));
		if (df_image != VX_DF_IMAGE_U8)
			status = VX_ERROR_INVALID_VALUE;
		vxReleaseImage(&image);
	}

	else if (index == 3)
	{
		vx_scalar scalar = 0; vx_enum type = 0;	vx_int32 value = 0;
		STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
		STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
		STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
		if (value < 0 || type != VX_TYPE_INT32)
			status = VX_ERROR_INVALID_VALUE;
		vxReleaseScalar(&scalar);
	}
	else if (index == 4)
	{
		vx_scalar scalar = 0; vx_enum type = 0;	vx_int32 value = 0;
		STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
		STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
		STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
		if (value < 0 || type != VX_TYPE_INT32)
			status = VX_ERROR_INVALID_VALUE;
		vxReleaseScalar(&scalar);
	}
	else if (index == 5)
	{
		vx_scalar scalar = 0; vx_enum type = 0;	vx_int32 value = 0;
		STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
		STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
		STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
		if (value < 0 || type != VX_TYPE_INT32)
			status = VX_ERROR_INVALID_VALUE;
		vxReleaseScalar(&scalar);
	}

	else if (index == 6)
	{
		vx_scalar scalar = 0; vx_enum type = 0;	vx_float32 value = 0;
		STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
		STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
		STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
		if (value < 0 || type != VX_TYPE_FLOAT32)
			status = VX_ERROR_INVALID_VALUE;
		vxReleaseScalar(&scalar);
	}

	else if (index == 7)
	{
		vx_scalar scalar = 0; vx_enum type = 0;	vx_float32 value = 0;
		STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
		STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
		STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
		if (value < 0 || type != VX_TYPE_FLOAT32)
			status = VX_ERROR_INVALID_VALUE;
		vxReleaseScalar(&scalar);
	}

	else if (index == 8)
	{
		vx_scalar scalar = 0; vx_enum type = 0;	vx_int32 value = 0;
		STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
		STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
		STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
		if (value < 0 || type != VX_TYPE_INT32)
			status = VX_ERROR_INVALID_VALUE;
		vxReleaseScalar(&scalar);
	}

	else if (index == 9)
	{
		vx_scalar scalar = 0; vx_enum type = 0;	vx_float32 value = 0;
		STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
		STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
		STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
		if (value < 0 || type != VX_TYPE_FLOAT32)
			status = VX_ERROR_INVALID_VALUE;
		vxReleaseScalar(&scalar);
	}

	else if (index == 10)
	{
		vx_scalar scalar = 0; vx_enum type = 0;	vx_float32 value = 0;
		STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
		STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
		STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
		if (value < 0 || type != VX_TYPE_FLOAT32)
			status = VX_ERROR_INVALID_VALUE;
		vxReleaseScalar(&scalar);
	}

	else if (index == 11)
	{
		vx_scalar scalar = 0; vx_enum type = 0;	vx_int32 value = 0;
		STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &scalar, sizeof(scalar)));
		STATUS_ERROR_CHECK(vxQueryScalar(scalar, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof(type)));
		STATUS_ERROR_CHECK(vxReadScalarValue(scalar, &value));
		if (value < 0 || type != VX_TYPE_INT32)
			status = VX_ERROR_INVALID_VALUE;
		vxReleaseScalar(&scalar);
	}

	vxReleaseParameter(&param);
	return status;
}

/*!***********************************************************************************************************
output parameter validator.
*************************************************************************************************************/
static vx_status VX_CALLBACK CV_MSER_feature_detector_OutputValidator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
	vx_status status = VX_SUCCESS;
	if (index == 1)
	{
		vx_parameter output_param = vxGetParameterByIndex(node, 1);
		vx_array output; vx_size size = 0;

		STATUS_ERROR_CHECK(vxQueryParameter(output_param, VX_PARAMETER_ATTRIBUTE_REF, &output, sizeof(vx_array)));
		STATUS_ERROR_CHECK(vxQueryArray(output, VX_ARRAY_ATTRIBUTE_CAPACITY, &size, sizeof(size)));

		if (size <= 0)
			status = VX_ERROR_INVALID_VALUE;

		vxReleaseArray(&output);
		vxReleaseParameter(&output_param);
	}
	return status;
}

/*!***********************************************************************************************************
Execution Kernel
*************************************************************************************************************/
static vx_status VX_CALLBACK CV_MSER_feature_detector_Kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
	vx_status status = VX_SUCCESS;

	vx_image image_in = (vx_image) parameters[0];
	vx_array array = (vx_array) parameters[1];
	vx_image mask = (vx_image) parameters[2];
	vx_scalar DELTA = (vx_scalar) parameters[3];
	vx_scalar MINAREA = (vx_scalar) parameters[4];
	vx_scalar MAXAREA = (vx_scalar) parameters[5];
	vx_scalar MAXVAR = (vx_scalar) parameters[6];
	vx_scalar MINDIV = (vx_scalar) parameters[7];
	vx_scalar MAXEVO = (vx_scalar) parameters[8];
	vx_scalar AREATHRES = (vx_scalar) parameters[9];
	vx_scalar MINMAR = (vx_scalar) parameters[10];
	vx_scalar EDGEBLUR = (vx_scalar) parameters[11];

	Mat *mat, *mask_mat, Img;
	vector<KeyPoint> key_points;
	int delta, min_area, max_area, max_evolution, edge_blur_size;
	float max_variation, min_diversity, area_threshold, min_margin;
	vx_float32 FloatValue = 0;
	vx_int32 value = 0;

	//Extracting Values from the Scalar 
	STATUS_ERROR_CHECK(vxReadScalarValue(MAXVAR, &FloatValue));	max_variation = FloatValue;
	STATUS_ERROR_CHECK(vxReadScalarValue(MINDIV, &FloatValue));	min_diversity = FloatValue;
	STATUS_ERROR_CHECK(vxReadScalarValue(AREATHRES, &FloatValue)); area_threshold = FloatValue;
	STATUS_ERROR_CHECK(vxReadScalarValue(MINMAR, &FloatValue));	min_margin = FloatValue;
	STATUS_ERROR_CHECK(vxReadScalarValue(DELTA, &value)); delta = value;
	STATUS_ERROR_CHECK(vxReadScalarValue(MINAREA, &value));	min_area = value;
	STATUS_ERROR_CHECK(vxReadScalarValue(MAXEVO, &value)); max_evolution = value;
	STATUS_ERROR_CHECK(vxReadScalarValue(MAXAREA, &value));	max_area = value;
	STATUS_ERROR_CHECK(vxReadScalarValue(EDGEBLUR, &value)); edge_blur_size = value;

	//Converting VX Image to OpenCV Mat
	STATUS_ERROR_CHECK(VX_to_CV_Image(&mat, image_in));
	STATUS_ERROR_CHECK(VX_to_CV_Image(&mask_mat, mask));

	//Compute using OpenCV
	Ptr<Feature2D> mser = MSER::create(delta, min_area, max_area, max_variation, min_diversity, max_evolution, area_threshold, min_margin, edge_blur_size);
	mser->detect(*mat, key_points, *mask_mat);

	//OpenCV 2.4.11 Call 
	//MSER MSER(delta, min_area, max_area, max_variation, min_diversity, max_evolution, area_threshold, min_margin, edge_blur_size);
	//MSER.detect(*mat, key_points, *mask_mat); ////Compute using OpenCV

	//Converting OpenCV Keypoints to OpenVX Keypoints
	STATUS_ERROR_CHECK(CV_to_VX_keypoints(key_points, array));


	return status;
}

/************************************************************************************************************
Function to Register the Kernel for Publish
*************************************************************************************************************/
vx_status CV_MSER_detect_Register(vx_context context)
{
	vx_status status = VX_SUCCESS;
	vx_kernel Kernel = vxAddKernel(context,
		"org.opencv.mser_detect",
		VX_KERNEL_OPENCV_MSER_DETECT,
		CV_MSER_feature_detector_Kernel,
		12,
		CV_MSER_feature_detector_InputValidator,
		CV_MSER_feature_detector_OutputValidator,
		nullptr,
		nullptr);

	if (Kernel)
	{
		PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 1, VX_BIDIRECTIONAL, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 2, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 3, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 4, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 5, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 6, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 7, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 8, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 9, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 10, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxAddParameterToKernel(Kernel, 11, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
		PARAM_ERROR_CHECK(vxFinalizeKernel(Kernel));
	}

	if (status != VX_SUCCESS)
	{
	exit:	vxRemoveKernel(Kernel); return VX_FAILURE;
	}

	return status;
}

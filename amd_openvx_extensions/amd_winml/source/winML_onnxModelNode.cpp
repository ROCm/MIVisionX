/*
Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.

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

// Node Global variables
string deviceName = "default";
LearningModel model = nullptr;
LearningModelDeviceKind deviceKind = LearningModelDeviceKind::Default;
LearningModelSession session = nullptr;
LearningModelBinding binding = nullptr;
VideoFrame imageFrame = nullptr;

/************************************************************************************************************
input parameter validator.
param [in] node The handle to the node.
param [in] index The index of the parameter to validate.
*************************************************************************************************************/
static vx_status VX_CALLBACK WINML_ImportOnnxModelAndRun_InputValidator(vx_node node, vx_uint32 index)
{
        vx_status status = VX_SUCCESS;
        vx_parameter param = vxGetParameterByIndex(node, index);

        if (index == 0 || index == 1 || index == 2)
        {
                vx_enum type;
                vx_scalar inputScalar;
                STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &inputScalar, sizeof(vx_scalar)));
                STATUS_ERROR_CHECK(vxQueryScalar(inputScalar, VX_SCALAR_TYPE, &type, sizeof(type)));
                if (type != VX_TYPE_STRING_AMD)
                {
                        printf("validate: ImportOnnxModelAndRun: #0 or #1 or #2 scalar type=%d (not VX_TYPE_STRING_AMD)\n", type);
                        return VX_ERROR_INVALID_TYPE;
                }
                vxReleaseScalar(&inputScalar);
        }
        else if (index == 3)
        {
                vx_size num_dims;
                vx_enum type;
                vx_tensor inputTensor;
                STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &inputTensor, sizeof(vx_tensor)));
                STATUS_ERROR_CHECK(vxQueryTensor(inputTensor, VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
                STATUS_ERROR_CHECK(vxQueryTensor(inputTensor, VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
                if (num_dims != 4)
                {
                        printf("validate: ImportOnnxModelAndRun: #3 num_dims=%zd (must be 4)\n", num_dims);
                        return VX_ERROR_INVALID_DIMENSION;
                }
                if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16))
                {
                        printf("validate: ImportOnnxModelAndRun: #3 tensor type=%d (not float/float16)\n", type);
                        return VX_ERROR_INVALID_TYPE;
                }
                vxReleaseTensor(&inputTensor);
        }

        vxReleaseParameter(&param);
        return status;
}

/************************************************************************************************************
output parameter validator.
param [in] node The handle to the node.
param [in] index The index of the parameter to validate.
param [out] meta
*************************************************************************************************************/
static vx_status VX_CALLBACK WINML_ImportOnnxModelAndRun_OutputValidator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
        vx_status status = VX_SUCCESS;
        vx_parameter param = vxGetParameterByIndex(node, index);

        if (index == 4)
        {
        vx_size num_dims;
        vx_enum type;
        vx_size output_dims_2[2], output_dims_4[4];
        vx_tensor outputTensor;
        STATUS_ERROR_CHECK(vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &outputTensor, sizeof(vx_tensor)));
        STATUS_ERROR_CHECK(vxQueryTensor(outputTensor, VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));
        STATUS_ERROR_CHECK(vxQueryTensor(outputTensor, VX_TENSOR_DATA_TYPE, &type, sizeof(type)));

        if (num_dims == 2)
        {
                STATUS_ERROR_CHECK(vxQueryTensor(outputTensor, VX_TENSOR_DIMS, output_dims_2, sizeof(output_dims_2)));
                STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(meta, VX_TENSOR_DIMS, output_dims_2, sizeof(output_dims_2)));
        }
        else if (num_dims == 4)
        {
                STATUS_ERROR_CHECK(vxQueryTensor(outputTensor, VX_TENSOR_DIMS, output_dims_4, sizeof(output_dims_4)))
                STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(meta, VX_TENSOR_DIMS, output_dims_4, sizeof(output_dims_4)))
        }
        else
        {
                printf("Output validate: ImportOnnxModelAndRun: #4 num_dims=%zd (must be 2/4)\n", num_dims);
                return VX_ERROR_INVALID_DIMENSION;
        }

        if ((type != VX_TYPE_FLOAT32) && (type != VX_TYPE_FLOAT16))
        {
                printf("Output validate: ImportOnnxModelAndRun: #4 tensor type=%d (not float/float16)\n", type);
                return VX_ERROR_INVALID_TYPE;
        }

        STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(meta, VX_TENSOR_DATA_TYPE, &type, sizeof(type)));
        STATUS_ERROR_CHECK(vxSetMetaFormatAttribute(meta, VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims)));

        vxReleaseTensor(&outputTensor);
        }
        return status;
}

/************************************************************************************************************
Initialize Kernel
param [in] node The handle to the node.
param [in] paramete.
param [in] num.
*************************************************************************************************************/
static vx_status VX_CALLBACK WINML_ImportOnnxModelAndRun_Initialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
        vx_status status = VX_SUCCESS;
		// Read scalar strings
		string modelLocation, inputName, outputName;
		vx_scalar modelLocationScalar = (vx_scalar)parameters[0];
		vx_scalar inputNameScalar = (vx_scalar)parameters[1];
		vx_scalar outputNameScalar = (vx_scalar)parameters[2];

		vxReadScalarValue(modelLocationScalar, &modelLocation);
		vxReadScalarValue(inputNameScalar, &inputName);
		vxReadScalarValue(outputNameScalar, &outputName);

		vxReleaseScalar(&modelLocationScalar);
		vxReleaseScalar(&inputNameScalar);
		vxReleaseScalar(&outputNameScalar);

        return status;
}

/************************************************************************************************************
Uninitialize Kernel
param [in] node The handle to the node.
param [in] paramete.
param [in] num.
*************************************************************************************************************/
static vx_status VX_CALLBACK WINML_ImportOnnxModelAndRun_Uninitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
        vx_status status = VX_SUCCESS;

        return status;
}

/************************************************************************************************************
Execution Kernel
param [in] node The handle to the node.
param [in] parameter.
param [in] num.
*************************************************************************************************************/
static vx_status VX_CALLBACK WINML_ImportOnnxModelAndRun_Kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
        vx_status status = VX_SUCCESS;

        return status;
}

/************************************************************************************************************
Function to Register the Kernel for Publish
*************************************************************************************************************/
vx_status  WINML_ImportOnnxModelAndRun_Register(vx_context context)
{
        vx_status status = VX_SUCCESS;
        vx_kernel kernel = vxAddKernel(context,
                "com.winml.import_onnx_model_and_run",
                VX_KERNEL_WINML_IMPORT_ONNX_MODEL_AND_RUN,
                WINML_ImportOnnxModelAndRun_Kernel,
                5,
                WINML_ImportOnnxModelAndRun_InputValidator,
                WINML_ImportOnnxModelAndRun_OutputValidator,
                WINML_ImportOnnxModelAndRun_Initialize,
                WINML_ImportOnnxModelAndRun_Uninitialize);

        if (kernel)
        {
                PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
                PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
                PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
                PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
                PARAM_ERROR_CHECK(vxAddParameterToKernel(kernel, 4, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
                PARAM_ERROR_CHECK(vxFinalizeKernel(kernel));
        }

        if (status != VX_SUCCESS)
        {
        exit:   vxRemoveKernel(kernel); return VX_FAILURE;
        }

        return status;
}

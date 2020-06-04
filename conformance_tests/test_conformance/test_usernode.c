/*
 * Copyright (c) 2012-2014 The Khronos Group Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 */

#include "test_engine/test.h"
#include <VX/vx.h>
#include <VX/vxu.h>

#define VX_KERNEL_CONFORMANCE_TEST_OWN_BAD (VX_KERNEL_BASE(VX_ID_MAX - 1, 0) + 0)
#define VX_KERNEL_CONFORMANCE_TEST_OWN_BAD_NAME "org.khronos.openvx.test.own_bad"

#define VX_KERNEL_CONFORMANCE_TEST_OWN (VX_KERNEL_BASE(VX_ID_MAX - 1, 0) + 1)
#define VX_KERNEL_CONFORMANCE_TEST_OWN_NAME "org.khronos.openvx.test.own"

TESTCASE(UserNode, CT_VXContext, ct_setup_vx_context, 0)

typedef enum _own_params_e {
    OWN_PARAM_INPUT = 0,
    OWN_PARAM_OUTPUT,
} own_params_e;

static vx_bool is_input_validator_called = vx_false_e;
static vx_status VX_CALLBACK own_InputValidator(vx_node node, vx_uint32 index)
{
    vx_status status = VX_ERROR_INVALID_PARAMETERS;
    vx_parameter param = 0;
    is_input_validator_called = vx_true_e;
    ASSERT_VX_OBJECT_(return VX_FAILURE, node, VX_TYPE_NODE);
    EXPECT(index == OWN_PARAM_INPUT);
    ASSERT_VX_OBJECT_(, param = vxGetParameterByIndex(node, index), VX_TYPE_PARAMETER);
    if (index == OWN_PARAM_INPUT)
    {
        vx_image image;
        vx_df_image df_image = 0;
        if (vxQueryParameter(param, VX_PARAMETER_ATTRIBUTE_REF, &image, sizeof(vx_image)) == VX_SUCCESS)
        {
            if(vxQueryImage(image, VX_IMAGE_ATTRIBUTE_FORMAT, &df_image, sizeof(df_image)) == VX_SUCCESS)
            {
                if (df_image == VX_DF_IMAGE_U8)
                    status = VX_SUCCESS;
                else
                    status = VX_ERROR_INVALID_VALUE;
            }
            vxReleaseImage(&image);
        }
    }
    vxReleaseParameter(&param);
    return status;
}

static vx_bool is_output_validator_called = vx_false_e;
static vx_status VX_CALLBACK own_OutputValidator(vx_node node, vx_uint32 index, vx_meta_format meta)
{
    vx_status status = VX_ERROR_INVALID_PARAMETERS;
    is_output_validator_called = vx_true_e;
    ASSERT_VX_OBJECT_(return VX_FAILURE, node, VX_TYPE_NODE);
    EXPECT(index == OWN_PARAM_OUTPUT);
    if (index == OWN_PARAM_OUTPUT)
    {
        vx_parameter in0 = vxGetParameterByIndex(node, OWN_PARAM_INPUT);
        vx_image input;
        if (vxQueryParameter(in0, VX_PARAMETER_ATTRIBUTE_REF, &input, sizeof(vx_image)) == VX_SUCCESS)
        {
            vx_uint32 width = 0, height = 0;
            vx_df_image format = VX_DF_IMAGE_VIRT;

            VX_CALL_(, vxQueryImage(input, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
            VX_CALL_(, vxQueryImage(input, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
            VX_CALL_(, vxQueryImage(input, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));

            VX_CALL_(, vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
            VX_CALL_(, vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
            VX_CALL_(, vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));

            vxReleaseImage(&input);

            status = VX_SUCCESS;
        }
        vxReleaseParameter(&in0);
    }
    return status;
}

static vx_bool is_kernel_called = vx_false_e;
static vx_status VX_CALLBACK own_Kernel(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    vx_status status = VX_ERROR_INVALID_PARAMETERS;
    is_kernel_called = vx_true_e;
    ASSERT_VX_OBJECT_(return VX_FAILURE, node, VX_TYPE_NODE);
    EXPECT(parameters != NULL);
    EXPECT(num == 2);
    if (num == 2)
    {
        vx_image input  = (vx_image)parameters[OWN_PARAM_INPUT];
        vx_image output = (vx_image)parameters[OWN_PARAM_OUTPUT];
        void *in = 0, *out = 0;
        vx_uint32 y, x;
        vx_imagepatch_addressing_t addr1, addr2;
        vx_rectangle_t rect;

        status = VX_SUCCESS;

        status |= vxGetValidRegionImage(input, &rect);
        status |= vxAccessImagePatch(input, &rect, 0, &addr1, &in, VX_READ_ONLY);
        status |= vxAccessImagePatch(output, &rect, 0, &addr2, &out, VX_WRITE_ONLY);
        for (y = 0; y < addr1.dim_y; y+=addr1.step_y)
        {
            for (x = 0; x < addr1.dim_x; x+=addr1.step_x)
            {
                // ...
            }
        }
        // write back and release
        status |= vxCommitImagePatch(output, &rect, 0, &addr2, out);
        status |= vxCommitImagePatch(input, NULL, 0, &addr1, in); // don't write back into the input
    }
    EXPECT_EQ_VX_STATUS(VX_SUCCESS, status);
    return status;
}

static vx_bool is_initialize_called = vx_false_e;
static vx_status VX_CALLBACK own_Initialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    is_initialize_called = vx_true_e;
    ASSERT_VX_OBJECT_(return VX_FAILURE, node, VX_TYPE_NODE);
    EXPECT(parameters != NULL);
    EXPECT(num == 2);
    if (parameters != NULL && num == 2)
    {
        EXPECT_VX_OBJECT(parameters[0], VX_TYPE_IMAGE);
        EXPECT_VX_OBJECT(parameters[1], VX_TYPE_IMAGE);
    }
    return VX_SUCCESS;
}

static vx_bool is_deinitialize_called = vx_false_e;
static vx_status VX_CALLBACK own_Deinitialize(vx_node node, const vx_reference *parameters, vx_uint32 num)
{
    is_deinitialize_called = vx_true_e;
    EXPECT(node != 0);
    EXPECT(parameters != NULL);
    EXPECT(num == 2);
    if (parameters != NULL && num == 2)
    {
        EXPECT_VX_OBJECT(parameters[0], VX_TYPE_IMAGE);
        EXPECT_VX_OBJECT(parameters[1], VX_TYPE_IMAGE);
    }
    return VX_SUCCESS;
}

static void own_register_kernel(vx_context context)
{
    vx_kernel kernel = 0;
    vx_size size = 0;

    ASSERT_VX_OBJECT(kernel = vxAddKernel(
            context,
            VX_KERNEL_CONFORMANCE_TEST_OWN_NAME,
            VX_KERNEL_CONFORMANCE_TEST_OWN,
            own_Kernel,
            2,
            own_InputValidator,
            own_OutputValidator,
            own_Initialize,
            own_Deinitialize), VX_TYPE_KERNEL);
    VX_CALL(vxAddParameterToKernel(kernel, OWN_PARAM_INPUT, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
    {
        vx_parameter parameter = 0;
        vx_enum direction = 0;
        ASSERT_VX_OBJECT(parameter = vxGetKernelParameterByIndex(kernel, OWN_PARAM_OUTPUT), VX_TYPE_PARAMETER);
        VX_CALL(vxQueryParameter(parameter, VX_PARAMETER_ATTRIBUTE_DIRECTION, &direction, sizeof(direction)));
        ASSERT(direction == VX_INPUT);
        VX_CALL(vxReleaseParameter(&parameter));
    }
    VX_CALL(vxAddParameterToKernel(kernel, OWN_PARAM_OUTPUT, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));
    VX_CALL(vxSetKernelAttribute(kernel, VX_KERNEL_ATTRIBUTE_LOCAL_DATA_SIZE, &size, sizeof(size)));
    VX_CALL(vxFinalizeKernel(kernel));
}


TEST(UserNode, testSimple)
{
    vx_context context = context_->vx_context_;
    vx_image src_image = 0, dst_image = 0;
    vx_graph graph = 0;
    vx_kernel kernel = 0;
    vx_node node = 0;

    is_input_validator_called = vx_false_e;
    is_output_validator_called = vx_false_e;
    is_kernel_called = vx_false_e;
    is_initialize_called = vx_false_e;
    is_deinitialize_called = vx_false_e;

    ASSERT_VX_OBJECT(src_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);
    ASSERT_VX_OBJECT(dst_image = vxCreateImage(context, 128, 128, VX_DF_IMAGE_U8), VX_TYPE_IMAGE);

    ASSERT_NO_FAILURE(own_register_kernel(context));

    ASSERT_VX_OBJECT(graph = vxCreateGraph(context), VX_TYPE_GRAPH);

    ASSERT_VX_OBJECT(kernel = vxGetKernelByName(context, VX_KERNEL_CONFORMANCE_TEST_OWN_NAME), VX_TYPE_KERNEL);
    ASSERT_VX_OBJECT(node = vxCreateGenericNode(graph, kernel), VX_TYPE_NODE);

    VX_CALL(vxSetParameterByIndex(node, 0, (vx_reference)src_image));
    VX_CALL(vxSetParameterByIndex(node, 1, (vx_reference)dst_image));

    VX_CALL(vxVerifyGraph(graph));
    VX_CALL(vxProcessGraph(graph));

    // We can't remove registered finalized kernel
    ASSERT_NE_VX_STATUS(VX_SUCCESS, vxRemoveKernel(kernel));

    vxReleaseNode(&node);
    vxReleaseKernel(&kernel);
    vxReleaseGraph(&graph);
    vxReleaseImage(&dst_image);
    vxReleaseImage(&src_image);

    ASSERT(node == 0);
    ASSERT(kernel == 0);
    ASSERT(graph == 0);
    ASSERT(dst_image == 0); ASSERT(src_image == 0);

    ASSERT(is_input_validator_called == vx_true_e);
    ASSERT(is_output_validator_called == vx_true_e);
    ASSERT(is_kernel_called == vx_true_e);
    ASSERT(is_initialize_called == vx_true_e);
    ASSERT(is_deinitialize_called == vx_true_e);
}


TEST(UserNode, testRemoveKernel)
{
    vx_context context = context_->vx_context_;
    vx_kernel kernel = 0;

    ASSERT_VX_OBJECT(kernel = vxAddKernel(
            context,
            VX_KERNEL_CONFORMANCE_TEST_OWN_BAD_NAME,
            VX_KERNEL_CONFORMANCE_TEST_OWN_BAD,
            own_Kernel,
            2,
            own_InputValidator,
            own_OutputValidator,
            own_Initialize,
            own_Deinitialize), VX_TYPE_KERNEL);

    VX_CALL(vxRemoveKernel(kernel));
#if 0 // TODO What specification says about this?
    VX_CALL(vxReleaseKernel(&kernel));
    ASSERT(kernel == 0);
#endif
}


TESTCASE_TESTS(UserNode,
        testSimple,
        testRemoveKernel
        )

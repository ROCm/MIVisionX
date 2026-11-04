/*
 * Copyright (c) 2012-2013 The Khronos Group Inc.
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

#include <VX/vx.h>
#include <VX/vxu.h>
#include <ago_internal.h>

static void vxuSetGraphAffinityDefault(vx_graph graph)
{
    vx_uint32 default_target = AGO_KERNEL_TARGET_DEFAULT;
    char textBuffer[1024];
    if (agoGetEnvironmentVariable("AGO_DEFAULT_TARGET", textBuffer, sizeof(textBuffer))) {
        if (!strcmp(textBuffer, "GPU")) {
            default_target = AGO_KERNEL_FLAG_DEVICE_GPU;
        } else if (!strcmp(textBuffer, "CPU")) {
            default_target = AGO_KERNEL_FLAG_DEVICE_CPU;
        }
    }
    graph->attr_affinity.device_type = default_target;
    graph->attr_affinity.device_info = 0;
}

 

VX_API_ENTRY vx_status VX_API_CALL vxuColorConvert(vx_context context, vx_image src, vx_image dst)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxColorConvertNode(graph, src, dst);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuChannelExtract(vx_context context, vx_image src, vx_enum channel, vx_image dst)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxChannelExtractNode(graph, src, channel, dst);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuChannelCombine(vx_context context,
                            vx_image plane0,
                            vx_image plane1,
                            vx_image plane2,
                            vx_image plane3,
                            vx_image output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxChannelCombineNode(graph, plane0, plane1, plane2, plane3, output);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

static vx_status vx_useImmediateBorderMode(vx_context context, vx_node node)
{
    vx_border_mode_t border;
    vx_status status = vxQueryContext(context, VX_CONTEXT_ATTRIBUTE_IMMEDIATE_BORDER_MODE, &border, sizeof(border));
    if (status == VX_SUCCESS)
        status = vxSetNodeAttribute(node, VX_NODE_ATTRIBUTE_BORDER_MODE, &border, sizeof(border));
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuSobel3x3(vx_context context, vx_image src, vx_image output_x, vx_image output_y)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxSobel3x3Node(graph, src, output_x, output_y);
        if (node)
        {
            status = vx_useImmediateBorderMode(context, node);
            if (status == VX_SUCCESS)
                status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuMagnitude(vx_context context, vx_image grad_x, vx_image grad_y, vx_image dst)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxMagnitudeNode(graph, grad_x, grad_y, dst);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuPhase(vx_context context, vx_image grad_x, vx_image grad_y, vx_image dst)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxPhaseNode(graph, grad_x, grad_y, dst);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuScaleImage(vx_context context, vx_image src, vx_image dst, vx_enum type)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxScaleImageNode(graph, src, dst, type);
        if (node)
        {
            status = vx_useImmediateBorderMode(context, node);
            if (status == VX_SUCCESS)
                status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuTableLookup(vx_context context, vx_image input, vx_lut lut, vx_image output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxTableLookupNode(graph, input, lut, output);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuHistogram(vx_context context, vx_image input, vx_distribution distribution)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxHistogramNode(graph, input, distribution);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuEqualizeHist(vx_context context, vx_image input, vx_image output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxEqualizeHistNode(graph, input, output);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuAbsDiff(vx_context context, vx_image in1, vx_image in2, vx_image out)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxAbsDiffNode(graph, in1, in2, out);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuMeanStdDev(vx_context context, vx_image input, vx_float32 *mean, vx_float32 *stddev)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_scalar s_mean = vxCreateScalar(context, VX_TYPE_FLOAT32, NULL);
        vx_scalar s_stddev = vxCreateScalar(context, VX_TYPE_FLOAT32, NULL);
        vx_node node = vxMeanStdDevNode(graph, input, s_mean, s_stddev);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
                if(mean) vxReadScalarValue(s_mean, mean);
                if(stddev) vxReadScalarValue(s_stddev, stddev);
            }
            vxReleaseNode(&node);
        }
        vxReleaseScalar(&s_mean);
        vxReleaseScalar(&s_stddev);
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuThreshold(vx_context context, vx_image input, vx_threshold thresh, vx_image output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxThresholdNode(graph, input, thresh, output);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuIntegralImage(vx_context context, vx_image input, vx_image output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxIntegralImageNode(graph, input, output);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuErode3x3(vx_context context, vx_image input, vx_image output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxErode3x3Node(graph, input, output);
        if (node)
        {
            status = vx_useImmediateBorderMode(context, node);
            if (status == VX_SUCCESS)
                status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuDilate3x3(vx_context context, vx_image input, vx_image output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxDilate3x3Node(graph, input, output);
        if (node)
        {
            status = vx_useImmediateBorderMode(context, node);
            if (status == VX_SUCCESS)
                status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuMedian3x3(vx_context context, vx_image input, vx_image output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxMedian3x3Node(graph, input, output);
        if (node)
        {
            status = vx_useImmediateBorderMode(context, node);
            if (status == VX_SUCCESS)
                status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuBox3x3(vx_context context, vx_image input, vx_image output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxBox3x3Node(graph, input, output);
        if (node)
        {
            status = vx_useImmediateBorderMode(context, node);
            if (status == VX_SUCCESS)
                status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuGaussian3x3(vx_context context, vx_image input, vx_image output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxGaussian3x3Node(graph, input, output);
        if (node)
        {
            status = vx_useImmediateBorderMode(context, node);
            if (status == VX_SUCCESS)
                status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuConvolve(vx_context context, vx_image input, vx_convolution conv, vx_image output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxConvolveNode(graph, input, conv, output);
        if (node)
        {
            status = vx_useImmediateBorderMode(context, node);
            if (status == VX_SUCCESS)
                status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuGaussianPyramid(vx_context context, vx_image input, vx_pyramid gaussian)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxGaussianPyramidNode(graph, input, gaussian);
        if (node)
        {
            status = vx_useImmediateBorderMode(context, node);
            if (status == VX_SUCCESS)
                status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuAccumulateImage(vx_context context, vx_image input, vx_image accum)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxAccumulateImageNode(graph, input, accum);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuAccumulateWeightedImage(vx_context context, vx_image input, vx_scalar scale, vx_image accum)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxAccumulateWeightedImageNode(graph, input, scale, accum);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuAccumulateSquareImage(vx_context context, vx_image input, vx_scalar scale, vx_image accum)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxAccumulateSquareImageNode(graph, input, scale, accum);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuMinMaxLoc(vx_context context, vx_image input,
                        vx_scalar minVal, vx_scalar maxVal,
                        vx_array minLoc, vx_array maxLoc,
                        vx_scalar minCount, vx_scalar maxCount)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxMinMaxLocNode(graph, input, minVal, maxVal, minLoc, maxLoc, minCount, maxCount);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuConvertDepth(vx_context context, vx_image input, vx_image output, vx_enum policy, vx_int32 shift)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    vx_scalar sshift = vxCreateScalar(context, VX_TYPE_INT32, &shift);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxConvertDepthNode(graph, input, output, policy, sshift);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    vxReleaseScalar(&sshift);
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuCannyEdgeDetector(vx_context context, vx_image input, vx_threshold hyst,
                               vx_int32 gradient_size, vx_enum norm_type,
                               vx_image output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxCannyEdgeDetectorNode(graph, input, hyst, gradient_size, norm_type, output);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuHalfScaleGaussian(vx_context context, vx_image input, vx_image output, vx_int32 kernel_size)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxHalfScaleGaussianNode(graph, input, output, kernel_size);
        if (node)
        {
            status = vx_useImmediateBorderMode(context, node);
            if (status == VX_SUCCESS)
                status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuAnd(vx_context context, vx_image in1, vx_image in2, vx_image out)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxAndNode(graph, in1, in2, out);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuOr(vx_context context, vx_image in1, vx_image in2, vx_image out)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxOrNode(graph, in1, in2, out);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuXor(vx_context context, vx_image in1, vx_image in2, vx_image out)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxXorNode(graph, in1, in2, out);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuNot(vx_context context, vx_image input, vx_image out)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxNotNode(graph, input, out);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuMultiply(vx_context context, vx_image in1, vx_image in2, vx_float32 scale, vx_enum overflow_policy, vx_enum rounding_policy, vx_image out)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    vx_scalar sscale = vxCreateScalar(context, VX_TYPE_FLOAT32, &scale);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxMultiplyNode(graph, in1, in2, sscale, overflow_policy, rounding_policy, out);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    vxReleaseScalar(&sscale);
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuAdd(vx_context context, vx_image in1, vx_image in2, vx_enum policy, vx_image out)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxAddNode(graph, in1, in2, policy, out);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuSubtract(vx_context context, vx_image in1, vx_image in2, vx_enum policy, vx_image out)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxSubtractNode(graph, in1, in2, policy, out);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuWarpAffine(vx_context context, vx_image input, vx_matrix matrix, vx_enum type, vx_image output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxWarpAffineNode(graph, input, matrix, type, output);
        if (node)
        {
            status = vx_useImmediateBorderMode(context, node);
            if (status == VX_SUCCESS)
                status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuWarpPerspective(vx_context context, vx_image input, vx_matrix matrix, vx_enum type, vx_image output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxWarpPerspectiveNode(graph, input, matrix, type, output);
        if (node)
        {
            status = vx_useImmediateBorderMode(context, node);
            if (status == VX_SUCCESS)
                status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuHarrisCorners(vx_context context, vx_image input,
        vx_scalar strength_thresh,
        vx_scalar min_distance,
        vx_scalar sensitivity,
        vx_int32 gradient_size,
        vx_int32 block_size,
        vx_array corners,
        vx_scalar num_corners)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxHarrisCornersNode(graph, input, strength_thresh, min_distance, sensitivity, gradient_size, block_size, corners, num_corners);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuFastCorners(vx_context context, vx_image input, vx_scalar sens, vx_bool nonmax, vx_array corners, vx_scalar num_corners)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxFastCornersNode(graph, input, sens, nonmax, corners, num_corners);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuOpticalFlowPyrLK(vx_context context, vx_pyramid old_images,
                              vx_pyramid new_images,
                              vx_array old_points,
                              vx_array new_points_estimates,
                              vx_array new_points,
                              vx_enum termination,
                              vx_scalar epsilon,
                              vx_scalar num_iterations,
                              vx_scalar use_initial_estimate,
                              vx_size window_dimension)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxOpticalFlowPyrLKNode(graph, old_images, new_images, old_points, new_points_estimates, new_points,
                termination,epsilon,num_iterations,use_initial_estimate,window_dimension);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuRemap(vx_context context, vx_image input, vx_remap table, vx_enum policy, vx_image output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxRemapNode(graph, input, table, policy, output);
        if (node)
        {
            status = vx_useImmediateBorderMode(context, node);
            if (status == VX_SUCCESS)
                status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuWeightedAverage(vx_context context, vx_image img1, vx_scalar alpha, vx_image img2, vx_image output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxWeightedAverageNode(graph, img1, alpha, img2, output);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuCopy(vx_context context, vx_reference input, vx_reference output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxCopyNode(graph, input, output);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuNonLinearFilter(vx_context context, vx_enum function, vx_image input, vx_matrix mask, vx_image output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxNonLinearFilterNode(graph, function, input, mask, output);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuLaplacianPyramid(vx_context context, vx_image input, vx_pyramid laplacian, vx_image output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxLaplacianPyramidNode(graph, input, laplacian, output);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuLaplacianReconstruct(vx_context context, vx_pyramid laplacian, vx_image input, vx_image output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
		vxuSetGraphAffinityDefault(graph);
		vx_node node = vxLaplacianReconstructNode(graph, laplacian, input, output);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
            {
                status = vxProcessGraph(graph);
            }
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

// #1790: immediate-mode wrappers for the vx_nodes.h functions added above --
// same create/verify/process/release pattern as every other vxu* function in
// this file. The underlying Node has no registered kernel for all of these
// except vxMin/vxMax/vxSelect-style simple kernels (none of which exist
// either, see vx_nodes.cpp), so vx*Node() returns NULL and these fall
// through to a clean VX_FAILURE.

VX_API_ENTRY vx_status VX_API_CALL vxuMin(vx_context context, vx_image in1, vx_image in2, vx_image out)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
        vxuSetGraphAffinityDefault(graph);
        vx_node node = vxMinNode(graph, in1, in2, out);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
                status = vxProcessGraph(graph);
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuMax(vx_context context, vx_image in1, vx_image in2, vx_image out)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
        vxuSetGraphAffinityDefault(graph);
        vx_node node = vxMaxNode(graph, in1, in2, out);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
                status = vxProcessGraph(graph);
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuNonMaxSuppression(vx_context context, vx_image input, vx_image mask, vx_int32 win_size, vx_image output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
        vxuSetGraphAffinityDefault(graph);
        vx_node node = vxNonMaxSuppressionNode(graph, input, mask, win_size, output);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
                status = vxProcessGraph(graph);
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuHOGCells(vx_context context, vx_image input, vx_int32 cell_width, vx_int32 cell_height, vx_int32 num_bins, vx_tensor magnitudes, vx_tensor bins)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
        vxuSetGraphAffinityDefault(graph);
        vx_node node = vxHOGCellsNode(graph, input, cell_width, cell_height, num_bins, magnitudes, bins);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
                status = vxProcessGraph(graph);
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuHOGFeatures(vx_context context, vx_image input, vx_tensor magnitudes, vx_tensor bins, const vx_hog_t *params, vx_size hog_param_size, vx_tensor features)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
        vxuSetGraphAffinityDefault(graph);
        vx_node node = vxHOGFeaturesNode(graph, input, magnitudes, bins, params, hog_param_size, features);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
                status = vxProcessGraph(graph);
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuHoughLinesP(vx_context context, vx_image input, const vx_hough_lines_p_t *params, vx_array lines_array, vx_scalar num_lines)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
        vxuSetGraphAffinityDefault(graph);
        vx_node node = vxHoughLinesPNode(graph, input, params, lines_array, num_lines);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
                status = vxProcessGraph(graph);
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuLBP(vx_context context, vx_image in, vx_enum format, vx_int8 kernel_size, vx_image out)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
        vxuSetGraphAffinityDefault(graph);
        vx_node node = vxLBPNode(graph, in, format, kernel_size, out);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
                status = vxProcessGraph(graph);
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuMatchTemplate(vx_context context, vx_image src, vx_image templateImage, vx_enum matchingMethod, vx_image output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
        vxuSetGraphAffinityDefault(graph);
        vx_node node = vxMatchTemplateNode(graph, src, templateImage, matchingMethod, output);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
                status = vxProcessGraph(graph);
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuBilateralFilter(vx_context context, vx_tensor src, vx_int32 diameter, vx_float32 sigmaSpace, vx_float32 sigmaValues, vx_tensor dst)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
        vxuSetGraphAffinityDefault(graph);
        vx_node node = vxBilateralFilterNode(graph, src, diameter, sigmaSpace, sigmaValues, dst);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
                status = vxProcessGraph(graph);
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuTensorMultiply(vx_context context, vx_tensor input1, vx_tensor input2, vx_scalar scale, vx_enum overflow_policy,
        vx_enum rounding_policy, vx_tensor output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
        vxuSetGraphAffinityDefault(graph);
        vx_node node = vxTensorMultiplyNode(graph, input1, input2, scale, overflow_policy, rounding_policy, output);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
                status = vxProcessGraph(graph);
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuTensorAdd(vx_context context, vx_tensor input1, vx_tensor input2, vx_enum policy, vx_tensor output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
        vxuSetGraphAffinityDefault(graph);
        vx_node node = vxTensorAddNode(graph, input1, input2, policy, output);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
                status = vxProcessGraph(graph);
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuTensorSubtract(vx_context context, vx_tensor input1, vx_tensor input2, vx_enum policy, vx_tensor output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
        vxuSetGraphAffinityDefault(graph);
        vx_node node = vxTensorSubtractNode(graph, input1, input2, policy, output);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
                status = vxProcessGraph(graph);
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuTensorTableLookup(vx_context context, vx_tensor input1, vx_lut lut, vx_tensor output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
        vxuSetGraphAffinityDefault(graph);
        vx_node node = vxTensorTableLookupNode(graph, input1, lut, output);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
                status = vxProcessGraph(graph);
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuTensorTranspose(vx_context context, vx_tensor input, vx_tensor output, vx_size dimension1, vx_size dimension2)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
        vxuSetGraphAffinityDefault(graph);
        vx_node node = vxTensorTransposeNode(graph, input, output, dimension1, dimension2);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
                status = vxProcessGraph(graph);
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuTensorConvertDepth(vx_context context, vx_tensor input, vx_enum policy, vx_scalar norm, vx_scalar offset, vx_tensor output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
        vxuSetGraphAffinityDefault(graph);
        vx_node node = vxTensorConvertDepthNode(graph, input, policy, norm, offset, output);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
                status = vxProcessGraph(graph);
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}

VX_API_ENTRY vx_status VX_API_CALL vxuTensorMatrixMultiply(vx_context context, vx_tensor input1, vx_tensor input2, vx_tensor input3,
    const vx_tensor_matrix_multiply_params_t *matrix_multiply_params, vx_tensor output)
{
    vx_status status = VX_FAILURE;
    vx_graph graph = vxCreateGraph(context);
    if (graph)
    {
        vxuSetGraphAffinityDefault(graph);
        vx_node node = vxTensorMatrixMultiplyNode(graph, input1, input2, input3, matrix_multiply_params, output);
        if (node)
        {
            status = vxVerifyGraph(graph);
            if (status == VX_SUCCESS)
                status = vxProcessGraph(graph);
            vxReleaseNode(&node);
        }
        vxReleaseGraph(&graph);
    }
    return status;
}
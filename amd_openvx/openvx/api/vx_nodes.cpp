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

/*!
 * \file
 * \brief The Graph Mode Interface for all Base Kernels.
 * \author Erik Rainey <erik.rainey@ti.com>
 */

#include "ago_internal.h"

static vx_node vxCreateNodeByStructure(vx_graph graph,
	vx_enum kernelenum,
	vx_reference params[],
	vx_uint32 num)
{
	vx_status status = VX_SUCCESS;
	vx_node node = 0;
	vx_context context = vxGetContext((vx_reference)graph);
	vx_kernel kernel = vxGetKernelByEnum(context, kernelenum);
	if (kernel)
	{
		node = vxCreateGenericNode(graph, kernel);
		if (node)
		{
			vx_uint32 p = 0;
			for (p = 0; p < num; p++)
			{
				if (params[p]) {
					status = vxSetParameterByIndex(node,
						p,
						params[p]);
					if (status != VX_SUCCESS)
					{
						vxAddLogEntry((vx_reference)graph, status, "Kernel %d Parameter %u is invalid.\n", kernelenum, p);
						vxReleaseNode(&node);
						node = 0;
						break;
					}
				}
			}
		}
		else
		{
			vxAddLogEntry((vx_reference)graph, VX_ERROR_INVALID_PARAMETERS, "Failed to create node with kernel enum %d\n", kernelenum);
			status = VX_ERROR_NO_MEMORY;
		}
		vxReleaseKernel(&kernel);
	}
	else
	{
		vxAddLogEntry((vx_reference)graph, VX_ERROR_INVALID_PARAMETERS, "failed to retrieve kernel enum %d\n", kernelenum);
		status = VX_ERROR_NOT_SUPPORTED;
	}
	return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxColorConvertNode(vx_graph graph, vx_image input, vx_image output)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph, VX_KERNEL_COLOR_CONVERT, params, dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxChannelExtractNode(vx_graph graph,
                             vx_image input,
                             vx_enum channelNum,
                             vx_image output)
{
    vx_context context = vxGetContext((vx_reference)graph);
    vx_scalar scalar = vxCreateScalar(context, VX_TYPE_ENUM, &channelNum);
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)scalar,
        (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_CHANNEL_EXTRACT,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&scalar); // node hold reference
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxChannelCombineNode(vx_graph graph,
                             vx_image plane0,
                             vx_image plane1,
                             vx_image plane2,
                             vx_image plane3,
                             vx_image output)
{
    vx_reference params[] = {
       (vx_reference)plane0,
       (vx_reference)plane1,
       (vx_reference)plane2,
       (vx_reference)plane3,
       (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_CHANNEL_COMBINE,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxSobel3x3Node(vx_graph graph, vx_image input, vx_image output_x, vx_image output_y)
{
    vx_reference params[] = {
       (vx_reference)input,
       (vx_reference)output_x,
       (vx_reference)output_y,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_SOBEL_3x3,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxMagnitudeNode(vx_graph graph, vx_image grad_x, vx_image grad_y, vx_image mag)
{
    vx_reference params[] = {
       (vx_reference)grad_x,
       (vx_reference)grad_y,
       (vx_reference)mag,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_MAGNITUDE,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxPhaseNode(vx_graph graph, vx_image grad_x, vx_image grad_y, vx_image orientation)
{
    vx_reference params[] = {
       (vx_reference)grad_x,
       (vx_reference)grad_y,
       (vx_reference)orientation,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_PHASE,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxScaleImageNode(vx_graph graph, vx_image src, vx_image dst, vx_enum type)
{
    vx_context context = vxGetContext((vx_reference)graph);
    vx_scalar stype = vxCreateScalar(context, VX_TYPE_ENUM, &type);
    vx_reference params[] = {
        (vx_reference)src,
        (vx_reference)dst,
        (vx_reference)stype,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_SCALE_IMAGE,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&stype);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxTableLookupNode(vx_graph graph, vx_image input, vx_lut lut, vx_image output)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)lut,
        (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_TABLE_LOOKUP,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxHistogramNode(vx_graph graph, vx_image input, vx_distribution distribution)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)distribution,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_HISTOGRAM,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxEqualizeHistNode(vx_graph graph, vx_image input, vx_image output)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_EQUALIZE_HISTOGRAM,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxAbsDiffNode(vx_graph graph, vx_image in1, vx_image in2, vx_image out)
{
    vx_reference params[] = {
       (vx_reference)in1,
       (vx_reference)in2,
       (vx_reference)out,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_ABSDIFF,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxMeanStdDevNode(vx_graph graph, vx_image input, vx_scalar mean, vx_scalar stddev)
{
    vx_reference params[] = {
       (vx_reference)input,
       (vx_reference)mean,
       (vx_reference)stddev,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_MEAN_STDDEV,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxThresholdNode(vx_graph graph, vx_image input, vx_threshold thesh, vx_image output)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)thesh,
        (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_THRESHOLD,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxIntegralImageNode(vx_graph graph, vx_image input, vx_image output)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_INTEGRAL_IMAGE,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxErode3x3Node(vx_graph graph, vx_image input, vx_image output)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_ERODE_3x3,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxDilate3x3Node(vx_graph graph, vx_image input, vx_image output)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_DILATE_3x3,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxMedian3x3Node(vx_graph graph, vx_image input, vx_image output)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_MEDIAN_3x3,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxBox3x3Node(vx_graph graph, vx_image input, vx_image output)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_BOX_3x3,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxGaussian3x3Node(vx_graph graph, vx_image input, vx_image output)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_GAUSSIAN_3x3,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxConvolveNode(vx_graph graph, vx_image input, vx_convolution conv, vx_image output)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)conv,
        (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_CUSTOM_CONVOLUTION,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxGaussianPyramidNode(vx_graph graph, vx_image input, vx_pyramid gaussian)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)gaussian,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_GAUSSIAN_PYRAMID,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxAccumulateImageNode(vx_graph graph, vx_image input, vx_image accum)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)accum,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_ACCUMULATE,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxAccumulateWeightedImageNode(vx_graph graph, vx_image input, vx_scalar alpha, vx_image accum)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)alpha,
        (vx_reference)accum,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_ACCUMULATE_WEIGHTED,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxAccumulateSquareImageNode(vx_graph graph, vx_image input, vx_scalar scalar, vx_image accum)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)scalar,
        (vx_reference)accum,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_ACCUMULATE_SQUARE,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxMinMaxLocNode(vx_graph graph,
                        vx_image input,
                        vx_scalar minVal, vx_scalar maxVal,
                        vx_array minLoc, vx_array maxLoc,
                        vx_scalar minCount, vx_scalar maxCount)
{
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)minVal,
        (vx_reference)maxVal,
        (vx_reference)minLoc,
        (vx_reference)maxLoc,
        (vx_reference)minCount,
        (vx_reference)maxCount,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_MINMAXLOC,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxConvertDepthNode(vx_graph graph, vx_image input, vx_image output, vx_enum policy, vx_scalar shift)
{
    vx_scalar pol = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_ENUM, &policy);
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)output,
        (vx_reference)pol,
        (vx_reference)shift,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                   VX_KERNEL_CONVERTDEPTH,
                                   params,
                                   dimof(params));
    vxReleaseScalar(&pol);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxCannyEdgeDetectorNode(vx_graph graph, vx_image input, vx_threshold hyst,
                                vx_int32 gradient_size, vx_enum norm_type,
                                vx_image output)
{
    vx_scalar gs = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &gradient_size);
    vx_scalar nt = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_ENUM, &norm_type);
    vx_reference params[] = {
        (vx_reference)input,
        (vx_reference)hyst,
        (vx_reference)gs,
        (vx_reference)nt,
        (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_CANNY_EDGE_DETECTOR,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&gs);
    vxReleaseScalar(&nt);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxAndNode(vx_graph graph, vx_image in1, vx_image in2, vx_image out)
{
    vx_reference params[] = {
       (vx_reference)in1,
       (vx_reference)in2,
       (vx_reference)out,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_AND,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxOrNode(vx_graph graph, vx_image in1, vx_image in2, vx_image out)
{
    vx_reference params[] = {
       (vx_reference)in1,
       (vx_reference)in2,
       (vx_reference)out,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_OR,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxXorNode(vx_graph graph, vx_image in1, vx_image in2, vx_image out)
{
    vx_reference params[] = {
       (vx_reference)in1,
       (vx_reference)in2,
       (vx_reference)out,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_XOR,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxNotNode(vx_graph graph, vx_image input, vx_image output)
{
    vx_reference params[] = {
       (vx_reference)input,
       (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph,
                                   VX_KERNEL_NOT,
                                   params,
                                   dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxMultiplyNode(vx_graph graph, vx_image in1, vx_image in2, vx_scalar scale, vx_enum overflow_policy, vx_enum rounding_policy, vx_image out)
{
    vx_context context = vxGetContext((vx_reference)graph);
    vx_scalar spolicy = vxCreateScalar(context, VX_TYPE_ENUM, &overflow_policy);
    vx_scalar rpolicy = vxCreateScalar(context, VX_TYPE_ENUM, &rounding_policy);
    vx_reference params[] = {
       (vx_reference)in1,
       (vx_reference)in2,
       (vx_reference)scale,
       (vx_reference)spolicy,
       (vx_reference)rpolicy,
       (vx_reference)out,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_MULTIPLY,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&spolicy);
    vxReleaseScalar(&rpolicy);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxAddNode(vx_graph graph, vx_image in1, vx_image in2, vx_enum policy, vx_image out)
{
    vx_context context = vxGetContext((vx_reference)graph);
    vx_scalar spolicy = vxCreateScalar(context, VX_TYPE_ENUM, &policy);
    vx_reference params[] = {
       (vx_reference)in1,
       (vx_reference)in2,
       (vx_reference)spolicy,
       (vx_reference)out,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_ADD,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&spolicy);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxSubtractNode(vx_graph graph, vx_image in1, vx_image in2, vx_enum policy, vx_image out)
{
    vx_context context = vxGetContext((vx_reference)graph);
    vx_scalar spolicy = vxCreateScalar(context, VX_TYPE_ENUM, &policy);
    vx_reference params[] = {
       (vx_reference)in1,
       (vx_reference)in2,
       (vx_reference)spolicy,
       (vx_reference)out,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_SUBTRACT,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&spolicy);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxWarpAffineNode(vx_graph graph, vx_image input, vx_matrix matrix, vx_enum type, vx_image output)
{
    vx_context context = vxGetContext((vx_reference)graph);
    vx_scalar stype = vxCreateScalar(context, VX_TYPE_ENUM, &type);
    vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)matrix,
            (vx_reference)stype,
            (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_WARP_AFFINE,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&stype);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxWarpPerspectiveNode(vx_graph graph, vx_image input, vx_matrix matrix, vx_enum type, vx_image output)
{
    vx_context context = vxGetContext((vx_reference)graph);
    vx_scalar stype = vxCreateScalar(context, VX_TYPE_ENUM, &type);
    vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)matrix,
            (vx_reference)stype,
            (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_WARP_PERSPECTIVE,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&stype);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxHarrisCornersNode(vx_graph graph,
                            vx_image input,
                            vx_scalar strength_thresh,
                            vx_scalar min_distance,
                            vx_scalar sensitivity,
                            vx_int32 gradient_size,
                            vx_int32 block_size,
                            vx_array corners,
                            vx_scalar num_corners)
{
    vx_scalar win = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &gradient_size);
    vx_scalar blk = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &block_size);
    vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)strength_thresh,
            (vx_reference)min_distance,
            (vx_reference)sensitivity,
            (vx_reference)win,
            (vx_reference)blk,
            (vx_reference)corners,
            (vx_reference)num_corners,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_HARRIS_CORNERS,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&win);
    vxReleaseScalar(&blk);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxFastCornersNode(vx_graph graph, vx_image input, vx_scalar strength_thresh, vx_bool nonmax_supression, vx_array corners, vx_scalar num_corners)
{
    vx_scalar nonmax = vxCreateScalar(vxGetContext((vx_reference)graph),VX_TYPE_BOOL, &nonmax_supression);
    vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)strength_thresh,
            (vx_reference)nonmax,
            (vx_reference)corners,
            (vx_reference)num_corners,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_FAST_CORNERS,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&nonmax);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxOpticalFlowPyrLKNode(vx_graph graph,
                               vx_pyramid old_images,
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
    vx_scalar term = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_ENUM, &termination);
    vx_scalar winsize = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_SIZE, &window_dimension);
    vx_reference params[] = {
            (vx_reference)old_images,
            (vx_reference)new_images,
            (vx_reference)old_points,
            (vx_reference)new_points_estimates,
            (vx_reference)new_points,
            (vx_reference)term,
            (vx_reference)epsilon,
            (vx_reference)num_iterations,
            (vx_reference)use_initial_estimate,
            (vx_reference)winsize,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_OPTICAL_FLOW_PYR_LK,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&term);
    vxReleaseScalar(&winsize);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxRemapNode(vx_graph graph,
                    vx_image input,
                    vx_remap table,
                    vx_enum policy,
                    vx_image output)
{
    vx_scalar spolicy = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_ENUM, &policy);
    vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)table,
            (vx_reference)spolicy,
            (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_REMAP,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&spolicy);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxHalfScaleGaussianNode(vx_graph graph, vx_image input, vx_image output, vx_int32 kernel_size)
{
    vx_scalar ksize = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &kernel_size);
    vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)output,
            (vx_reference)ksize,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_HALFSCALE_GAUSSIAN,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&ksize);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxCopyNode(vx_graph graph, vx_reference input, vx_reference output)
{
    vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_COPY,
                                           params,
                                           dimof(params));
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxWeightedAverageNode(vx_graph graph, vx_image img1, vx_scalar alpha, vx_image img2, vx_image output)
{
    vx_reference params[] = {
            (vx_reference)img1,
            (vx_reference)alpha,
            (vx_reference)img2,
            (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_WEIGHTED_AVERAGE,
                                           params,
                                           dimof(params));
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxNonLinearFilterNode(vx_graph graph, vx_enum function, vx_image input, vx_matrix mask, vx_image output)
{
    vx_scalar func = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_ENUM, &function);
    vx_reference params[] = {
            (vx_reference)func,
            (vx_reference)input,
            (vx_reference)mask,
            (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_NON_LINEAR_FILTER,
                                           params,
                                           dimof(params));
    vxReleaseScalar(&func);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxLaplacianPyramidNode(vx_graph graph, vx_image input, vx_pyramid laplacian, vx_image output)
{
    vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)laplacian,
            (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_LAPLACIAN_PYRAMID,
                                           params,
                                           dimof(params));
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxLaplacianReconstructNode(vx_graph graph, vx_pyramid laplacian, vx_image input, vx_image output)
{
    vx_reference params[] = {
            (vx_reference)laplacian,
            (vx_reference)input,
            (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph,
                                           VX_KERNEL_LAPLACIAN_RECONSTRUCT,
                                           params,
                                           dimof(params));
    return node;
}

// #1790: the following functions are declared in the installed headers but
// have no backing kernel registered in ago_kernel_list.cpp (vxSelectNode is
// the sole exception -- its kernel exists but is itself still a stub, see
// #1791/agoKernel_Select_DATA_DATA_DATA). vxCreateNodeByStructure already
// returns a clean NULL node when vxGetKernelByEnum finds no match, so these
// thin wrappers compile, link, and fail cleanly at verify/process time
// without needing any new kernel implementation.

VX_API_ENTRY vx_node VX_API_CALL vxMinNode(vx_graph graph, vx_image in1, vx_image in2, vx_image out)
{
    vx_reference params[] = {
            (vx_reference)in1,
            (vx_reference)in2,
            (vx_reference)out,
    };
    return vxCreateNodeByStructure(graph, VX_KERNEL_MIN, params, dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxMaxNode(vx_graph graph, vx_image in1, vx_image in2, vx_image out)
{
    vx_reference params[] = {
            (vx_reference)in1,
            (vx_reference)in2,
            (vx_reference)out,
    };
    return vxCreateNodeByStructure(graph, VX_KERNEL_MAX, params, dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxSelectNode(vx_graph graph, vx_scalar condition, vx_reference true_value, vx_reference false_value, vx_reference output)
{
    vx_reference params[] = {
            (vx_reference)condition,
            true_value,
            false_value,
            output,
    };
    return vxCreateNodeByStructure(graph, VX_KERNEL_SELECT, params, dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxScalarOperationNode(vx_graph graph, vx_enum scalar_operation, vx_scalar a, vx_scalar b, vx_scalar output)
{
    vx_scalar sop = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_ENUM, &scalar_operation);
    vx_reference params[] = {
            (vx_reference)sop,
            (vx_reference)a,
            (vx_reference)b,
            (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph, VX_KERNEL_SCALAR_OPERATION, params, dimof(params));
    vxReleaseScalar(&sop);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxNonMaxSuppressionNode(vx_graph graph, vx_image input, vx_image mask, vx_int32 win_size, vx_image output)
{
    vx_scalar winsize = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &win_size);
    vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)mask,
            (vx_reference)winsize,
            (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph, VX_KERNEL_NON_MAX_SUPPRESSION, params, dimof(params));
    vxReleaseScalar(&winsize);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxHOGCellsNode(vx_graph graph, vx_image input, vx_int32 cell_width, vx_int32 cell_height, vx_int32 num_bins, vx_tensor magnitudes, vx_tensor bins)
{
    vx_scalar cw = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &cell_width);
    vx_scalar ch = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &cell_height);
    vx_scalar nb = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &num_bins);
    vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)cw,
            (vx_reference)ch,
            (vx_reference)nb,
            (vx_reference)magnitudes,
            (vx_reference)bins,
    };
    vx_node node = vxCreateNodeByStructure(graph, VX_KERNEL_HOG_CELLS, params, dimof(params));
    vxReleaseScalar(&cw);
    vxReleaseScalar(&ch);
    vxReleaseScalar(&nb);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxHOGFeaturesNode(vx_graph graph, vx_image input, vx_tensor magnitudes, vx_tensor bins, const vx_hog_t *params_struct, vx_size hog_param_size, vx_tensor features)
{
    // no VX_KERNEL_HOG_FEATURES kernel is registered, so this always fails
    // cleanly regardless of how params_struct would be wrapped; omitted here.
    (void)params_struct;
    (void)hog_param_size;
    vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)magnitudes,
            (vx_reference)bins,
            (vx_reference)features,
    };
    return vxCreateNodeByStructure(graph, VX_KERNEL_HOG_FEATURES, params, dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxHoughLinesPNode(vx_graph graph, vx_image input, const vx_hough_lines_p_t *params_struct, vx_array lines_array, vx_scalar num_lines)
{
    // no VX_KERNEL_HOUGH_LINES_P kernel is registered; see vxHOGFeaturesNode.
    (void)params_struct;
    vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)lines_array,
            (vx_reference)num_lines,
    };
    return vxCreateNodeByStructure(graph, VX_KERNEL_HOUGH_LINES_P, params, dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxLBPNode(vx_graph graph, vx_image in, vx_enum format, vx_int8 kernel_size, vx_image out)
{
    vx_scalar fmt = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_ENUM, &format);
    vx_scalar ksize = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT8, &kernel_size);
    vx_reference params[] = {
            (vx_reference)in,
            (vx_reference)fmt,
            (vx_reference)ksize,
            (vx_reference)out,
    };
    vx_node node = vxCreateNodeByStructure(graph, VX_KERNEL_LBP, params, dimof(params));
    vxReleaseScalar(&fmt);
    vxReleaseScalar(&ksize);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxMatchTemplateNode(vx_graph graph, vx_image src, vx_image templateImage, vx_enum matchingMethod, vx_image output)
{
    vx_scalar method = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_ENUM, &matchingMethod);
    vx_reference params[] = {
            (vx_reference)src,
            (vx_reference)templateImage,
            (vx_reference)method,
            (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph, VX_KERNEL_MATCH_TEMPLATE, params, dimof(params));
    vxReleaseScalar(&method);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxBilateralFilterNode(vx_graph graph, vx_tensor src, vx_int32 diameter, vx_float32 sigmaSpace, vx_float32 sigmaValues, vx_tensor dst)
{
    vx_scalar d = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &diameter);
    vx_scalar ss = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &sigmaSpace);
    vx_scalar sv = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &sigmaValues);
    vx_reference params[] = {
            (vx_reference)src,
            (vx_reference)d,
            (vx_reference)ss,
            (vx_reference)sv,
            (vx_reference)dst,
    };
    vx_node node = vxCreateNodeByStructure(graph, VX_KERNEL_BILATERAL_FILTER, params, dimof(params));
    vxReleaseScalar(&d);
    vxReleaseScalar(&ss);
    vxReleaseScalar(&sv);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxTensorMultiplyNode(vx_graph graph, vx_tensor input1, vx_tensor input2, vx_scalar scale, vx_enum overflow_policy, vx_enum rounding_policy, vx_tensor output)
{
    vx_scalar op = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_ENUM, &overflow_policy);
    vx_scalar rp = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_ENUM, &rounding_policy);
    vx_reference params[] = {
            (vx_reference)input1,
            (vx_reference)input2,
            (vx_reference)scale,
            (vx_reference)op,
            (vx_reference)rp,
            (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph, VX_KERNEL_TENSOR_MULTIPLY, params, dimof(params));
    vxReleaseScalar(&op);
    vxReleaseScalar(&rp);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxTensorAddNode(vx_graph graph, vx_tensor input1, vx_tensor input2, vx_enum policy, vx_tensor output)
{
    vx_scalar pol = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_ENUM, &policy);
    vx_reference params[] = {
            (vx_reference)input1,
            (vx_reference)input2,
            (vx_reference)pol,
            (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph, VX_KERNEL_TENSOR_ADD, params, dimof(params));
    vxReleaseScalar(&pol);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxTensorSubtractNode(vx_graph graph, vx_tensor input1, vx_tensor input2, vx_enum policy, vx_tensor output)
{
    vx_scalar pol = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_ENUM, &policy);
    vx_reference params[] = {
            (vx_reference)input1,
            (vx_reference)input2,
            (vx_reference)pol,
            (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph, VX_KERNEL_TENSOR_SUBTRACT, params, dimof(params));
    vxReleaseScalar(&pol);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxTensorTableLookupNode(vx_graph graph, vx_tensor input1, vx_lut lut, vx_tensor output)
{
    vx_reference params[] = {
            (vx_reference)input1,
            (vx_reference)lut,
            (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph, VX_KERNEL_TENSOR_TABLE_LOOKUP, params, dimof(params));
}

VX_API_ENTRY vx_node VX_API_CALL vxTensorTransposeNode(vx_graph graph, vx_tensor input, vx_tensor output, vx_size dimension1, vx_size dimension2)
{
    vx_scalar d1 = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_SIZE, &dimension1);
    vx_scalar d2 = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_SIZE, &dimension2);
    vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)output,
            (vx_reference)d1,
            (vx_reference)d2,
    };
    vx_node node = vxCreateNodeByStructure(graph, VX_KERNEL_TENSOR_TRANSPOSE, params, dimof(params));
    vxReleaseScalar(&d1);
    vxReleaseScalar(&d2);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxTensorConvertDepthNode(vx_graph graph, vx_tensor input, vx_enum policy, vx_scalar norm, vx_scalar offset, vx_tensor output)
{
    vx_scalar pol = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_ENUM, &policy);
    vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)pol,
            (vx_reference)norm,
            (vx_reference)offset,
            (vx_reference)output,
    };
    vx_node node = vxCreateNodeByStructure(graph, VX_KERNEL_TENSOR_CONVERT_DEPTH, params, dimof(params));
    vxReleaseScalar(&pol);
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxTensorMatrixMultiplyNode(vx_graph graph, vx_tensor input1, vx_tensor input2, vx_tensor input3,
    const vx_tensor_matrix_multiply_params_t *matrix_multiply_params, vx_tensor output)
{
    // no VX_KERNEL_TENSOR_MATRIX_MULTIPLY kernel is registered; see vxHOGFeaturesNode.
    (void)matrix_multiply_params;
    vx_reference params[] = {
            (vx_reference)input1,
            (vx_reference)input2,
            (vx_reference)input3,
            (vx_reference)output,
    };
    return vxCreateNodeByStructure(graph, VX_KERNEL_TENSOR_MATRIX_MULTIPLY, params, dimof(params));
}

// vxNormalizationLayer (VX/vx_compatibility.h) is a legacy-named alias with
// no dedicated kernel enum of its own; it maps most closely to the NN
// extension's local response normalization, which also has no registered
// kernel, so this fails cleanly the same way as the functions above.
VX_API_ENTRY vx_node VX_API_CALL vxNormalizationLayer(vx_graph graph, vx_tensor inputs, vx_enum type,
        vx_size normalization_size, vx_float32 alpha, vx_float32 beta, vx_tensor outputs)
{
    vx_scalar styp = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_ENUM, &type);
    vx_scalar snorm = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_SIZE, &normalization_size);
    vx_scalar salpha = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &alpha);
    vx_scalar sbeta = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &beta);
    vx_reference params[] = {
            (vx_reference)inputs,
            (vx_reference)styp,
            (vx_reference)snorm,
            (vx_reference)salpha,
            (vx_reference)sbeta,
            (vx_reference)outputs,
    };
    vx_node node = vxCreateNodeByStructure(graph, VX_KERNEL_LOCAL_RESPONSE_NORMALIZATION_LAYER, params, dimof(params));
    vxReleaseScalar(&styp);
    vxReleaseScalar(&snorm);
    vxReleaseScalar(&salpha);
    vxReleaseScalar(&sbeta);
    return node;
}
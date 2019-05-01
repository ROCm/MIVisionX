/* 
Copyright (c) 2015 Advanced Micro Devices, Inc. All rights reserved.
 
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


#include "ago_internal.h"
#include "ago_kernel_api.h"
#include "ago_haf_gpu.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Local Utility Functions
//
static int ValidateArguments_Img_1IN(AgoNode * node, vx_df_image fmtIn)
{
	// validate parameters
	vx_uint32 width = node->paramList[1]->u.img.width;
	vx_uint32 height = node->paramList[1]->u.img.height;
	if (node->paramList[1]->u.img.format != fmtIn)
		return VX_ERROR_INVALID_FORMAT;
	else if (!width || !height)
		return VX_ERROR_INVALID_DIMENSION;
	return VX_SUCCESS;
}
static int ValidateArguments_Img_1OUT(AgoNode * node, vx_df_image fmtOut)
{
	// validate parameters
	vx_uint32 width = node->paramList[0]->u.img.width;
	vx_uint32 height = node->paramList[0]->u.img.height;
	if (node->paramList[0]->u.img.format != fmtOut)
		return VX_ERROR_INVALID_FORMAT;
	else if (!width || !height)
		return VX_ERROR_INVALID_DIMENSION;
	// set output image sizes are same as input image size
	vx_meta_format meta;
	meta = &node->metaList[0];
	meta->data.u.img.width = width;
	meta->data.u.img.height = height;
	meta->data.u.img.format = fmtOut;
	return VX_SUCCESS;
}
static int ValidateArguments_Img_1OUT_1IN(AgoNode * node, vx_df_image fmtOut, vx_df_image fmtIn, bool bShrinkValidRegion = false, int shrinkValidRegion_x = 0, int shrinkValidRegion_y = 0)
{
	// validate parameters
	vx_uint32 width = node->paramList[1]->u.img.width;
	vx_uint32 height = node->paramList[1]->u.img.height;
	if (node->paramList[1]->u.img.format != fmtIn)
		return VX_ERROR_INVALID_FORMAT;
	else if (!width || !height)
		return VX_ERROR_INVALID_DIMENSION;
	// set output image sizes are same as input image size
	vx_meta_format meta;
	meta = &node->metaList[0];
	meta->data.u.img.width = width;
	meta->data.u.img.height = height;
	meta->data.u.img.format = fmtOut;
	return VX_SUCCESS;
}
static int ValidateArguments_Img_2OUT_1IN(AgoNode * node, vx_df_image fmtOut1, vx_df_image fmtOut2, vx_df_image fmtIn, bool bShrinkValidRegion = false, int shrinkValidRegion_x = 0, int shrinkValidRegion_y = 0)
{
	// validate parameters
	vx_uint32 width = node->paramList[2]->u.img.width;
	vx_uint32 height = node->paramList[2]->u.img.height;
	if (node->paramList[2]->u.img.format != fmtIn)
		return VX_ERROR_INVALID_FORMAT;
	else if (!width || !height)
		return VX_ERROR_INVALID_DIMENSION;
	// set output image sizes are same as input image size
	vx_meta_format meta;
	meta = &node->metaList[0];
	meta->data.u.img.width = width;
	meta->data.u.img.height = height;
	meta->data.u.img.format = fmtOut1;
	meta = &node->metaList[1];
	meta->data.u.img.width = width;
	meta->data.u.img.height = height;
	meta->data.u.img.format = fmtOut2;
	return VX_SUCCESS;
}
static int ValidateArguments_Img_3OUT_1IN(AgoNode * node, vx_df_image fmtOut1, vx_df_image fmtOut2, vx_df_image fmtOut3, vx_df_image fmtIn, bool bShrinkValidRegion = false, int shrinkValidRegion_x = 0, int shrinkValidRegion_y = 0)
{
	// validate parameters
	vx_uint32 width = node->paramList[3]->u.img.width;
	vx_uint32 height = node->paramList[3]->u.img.height;
	if (node->paramList[3]->u.img.format != fmtIn)
		return VX_ERROR_INVALID_FORMAT;
	else if (!width || !height)
		return VX_ERROR_INVALID_DIMENSION;
	// set output image sizes are same as input image size
	vx_meta_format meta;
	meta = &node->metaList[0];
	meta->data.u.img.width = width;
	meta->data.u.img.height = height;
	meta->data.u.img.format = fmtOut1;
	meta = &node->metaList[1];
	meta->data.u.img.width = width;
	meta->data.u.img.height = height;
	meta->data.u.img.format = fmtOut2;
	meta = &node->metaList[2];
	meta->data.u.img.width = width;
	meta->data.u.img.height = height;
	meta->data.u.img.format = fmtOut3;
	return VX_SUCCESS;
}
static int ValidateArguments_Img_4OUT_1IN(AgoNode * node, vx_df_image fmtOut1, vx_df_image fmtOut2, vx_df_image fmtOut3, vx_df_image fmtOut4, vx_df_image fmtIn, bool bShrinkValidRegion = false, int shrinkValidRegion_x = 0, int shrinkValidRegion_y = 0)
{
	// validate parameters
	vx_uint32 width = node->paramList[4]->u.img.width;
	vx_uint32 height = node->paramList[4]->u.img.height;
	if (node->paramList[4]->u.img.format != fmtIn)
		return VX_ERROR_INVALID_FORMAT;
	else if (!width || !height)
		return VX_ERROR_INVALID_DIMENSION;
	// set output image sizes are same as input image size
	vx_meta_format meta;
	meta = &node->metaList[0];
	meta->data.u.img.width = width;
	meta->data.u.img.height = height;
	meta->data.u.img.format = fmtOut1;
	meta = &node->metaList[1];
	meta->data.u.img.width = width;
	meta->data.u.img.height = height;
	meta->data.u.img.format = fmtOut2;
	meta = &node->metaList[2];
	meta->data.u.img.width = width;
	meta->data.u.img.height = height;
	meta->data.u.img.format = fmtOut3;
	meta = &node->metaList[3];
	meta->data.u.img.width = width;
	meta->data.u.img.height = height;
	meta->data.u.img.format = fmtOut4;
	return VX_SUCCESS;
}
static int ValidateArguments_Img_1OUT_1IN_S(AgoNode * node, vx_df_image fmtOut, vx_df_image fmtIn, vx_enum scalarType, bool bShrinkValidRegion = false, int shrinkValidRegion_x = 0, int shrinkValidRegion_y = 0)
{
	int status = ValidateArguments_Img_1OUT_1IN(node, fmtOut, fmtIn, bShrinkValidRegion, shrinkValidRegion_x, shrinkValidRegion_y);
	if (!status) {
		if (node->paramList[2]->u.scalar.type != scalarType)
			return VX_ERROR_INVALID_TYPE;
	}
	return status;
}
static int ValidateArguments_Img_1OUT_1IN_2S(AgoNode * node, vx_df_image fmtOut, vx_df_image fmtIn, vx_enum scalarType, vx_enum scalarType2, bool bShrinkValidRegion = false, int shrinkValidRegion_x = 0, int shrinkValidRegion_y = 0)
{
	int status = ValidateArguments_Img_1OUT_1IN(node, fmtOut, fmtIn, bShrinkValidRegion, shrinkValidRegion_x, shrinkValidRegion_y);
	if (!status) {
		if (node->paramList[2]->u.scalar.type != scalarType && node->paramList[3]->u.scalar.type != scalarType2)
			return VX_ERROR_INVALID_TYPE;
	}
	return status;
}
static int ValidateArguments_Img_1OUT_1IN_3S(AgoNode * node, vx_df_image fmtOut, vx_df_image fmtIn, vx_enum scalarType, vx_enum scalarType2, vx_enum scalarType3, bool bShrinkValidRegion = false, int shrinkValidRegion_x = 0, int shrinkValidRegion_y = 0)
{
	int status = ValidateArguments_Img_1OUT_1IN(node, fmtOut, fmtIn, bShrinkValidRegion, shrinkValidRegion_x, shrinkValidRegion_y);
	if (!status) {
		if (node->paramList[2]->u.scalar.type != scalarType && node->paramList[3]->u.scalar.type != scalarType2 && node->paramList[4]->u.scalar.type != scalarType3)
			return VX_ERROR_INVALID_TYPE;
	}
	return status;
}
static int ValidateArguments_Img_1OUT_2IN(AgoNode * node, vx_df_image fmtOut, vx_df_image fmtIn1, vx_df_image fmtIn2)
{
	// validate parameters
	vx_uint32 width = node->paramList[1]->u.img.width;
	vx_uint32 height = node->paramList[1]->u.img.height;
	if (node->paramList[1]->u.img.format != fmtIn1 || node->paramList[2]->u.img.format != fmtIn2)
		return VX_ERROR_INVALID_FORMAT;
	else if (!width || !height || width != node->paramList[2]->u.img.width || height != node->paramList[2]->u.img.height)
		return VX_ERROR_INVALID_DIMENSION;
	// set output image sizes are same as input image size
	vx_meta_format meta;
	meta = &node->metaList[0];
	meta->data.u.img.width = width;
	meta->data.u.img.height = height;
	meta->data.u.img.format = fmtOut;
	return VX_SUCCESS;
}
static int ValidateArguments_Img_1OUT_2IN_S(AgoNode * node, vx_df_image fmtOut, vx_df_image fmtIn1, vx_df_image fmtIn2, vx_enum scalarType)
{
	int status = ValidateArguments_Img_1OUT_2IN(node, fmtOut, fmtIn1, fmtIn2);
	if (!status) {
		if (node->paramList[3]->u.scalar.type != scalarType)
			return VX_ERROR_INVALID_TYPE;
	}
	return status;
}
static int ValidateArguments_Img_1OUT_3IN(AgoNode * node, vx_df_image fmtOut, vx_df_image fmtIn1, vx_df_image fmtIn2, vx_df_image fmtIn3)
{
	// validate parameters
	vx_uint32 width = node->paramList[1]->u.img.width;
	vx_uint32 height = node->paramList[1]->u.img.height;
	if (node->paramList[1]->u.img.format != fmtIn1 || node->paramList[2]->u.img.format != fmtIn2 || 
		node->paramList[3]->u.img.format != fmtIn3)
		return VX_ERROR_INVALID_FORMAT;
	else if (!width || !height || width != node->paramList[2]->u.img.width || height != node->paramList[2]->u.img.height || 
		                          width != node->paramList[3]->u.img.width || height != node->paramList[3]->u.img.height)
		return VX_ERROR_INVALID_DIMENSION;
	// set output image sizes are same as input image size
	vx_meta_format meta;
	meta = &node->metaList[0];
	meta->data.u.img.width = width;
	meta->data.u.img.height = height;
	meta->data.u.img.format = fmtOut;
	return VX_SUCCESS;
}
static int ValidateArguments_Img_1OUT_4IN(AgoNode * node, vx_df_image fmtOut, vx_df_image fmtIn1, vx_df_image fmtIn2, vx_df_image fmtIn3, vx_df_image fmtIn4)
{
	// validate parameters
	vx_uint32 width = node->paramList[1]->u.img.width;
	vx_uint32 height = node->paramList[1]->u.img.height;
	if (node->paramList[1]->u.img.format != fmtIn1 || node->paramList[2]->u.img.format != fmtIn2 ||
		node->paramList[3]->u.img.format != fmtIn3 || node->paramList[4]->u.img.format != fmtIn4)
		return VX_ERROR_INVALID_FORMAT;
	else if (!width || !height || width != node->paramList[2]->u.img.width || height != node->paramList[2]->u.img.height ||
								  width != node->paramList[3]->u.img.width || height != node->paramList[3]->u.img.height ||
								  width != node->paramList[4]->u.img.width || height != node->paramList[4]->u.img.height)
		return VX_ERROR_INVALID_DIMENSION;
	// set output image sizes are same as input image size
	vx_meta_format meta;
	meta = &node->metaList[0];
	meta->data.u.img.width = width;
	meta->data.u.img.height = height;
	meta->data.u.img.format = fmtOut;
	return VX_SUCCESS;
}
static int ValidateArguments_CannySuppThreshold_U8(AgoNode * node, vx_df_image fmtIn, int shrinkValidRegion_x, int shrinkValidRegion_y)
{
	// validate parameters
	vx_uint32 width = node->paramList[1]->u.img.width;
	vx_uint32 height = node->paramList[1]->u.img.height;
	if (node->paramList[1]->u.img.format != fmtIn)
		return VX_ERROR_INVALID_FORMAT;
	else if (!width || !height)
		return VX_ERROR_INVALID_DIMENSION;
	if (node->paramList[2]->u.thr.thresh_type != VX_THRESHOLD_TYPE_RANGE ||
		(node->paramList[2]->u.thr.data_type != VX_TYPE_UINT8 && node->paramList[2]->u.thr.data_type != VX_TYPE_UINT16 && node->paramList[2]->u.thr.data_type != VX_TYPE_INT16))
		return VX_ERROR_INVALID_TYPE;
	// set output info
	vx_meta_format meta;
	meta = &node->metaList[0];
	meta->data.u.img.width = width;
	meta->data.u.img.height = height;
	meta->data.u.img.format = VX_DF_IMAGE_U8;
	return VX_SUCCESS;
}
static int ValidateArguments_CannySuppThreshold_U8XY(AgoNode * node, vx_df_image fmtIn, int shrinkValidRegion_x, int shrinkValidRegion_y)
{
	// validate parameters
	vx_uint32 width = node->paramList[2]->u.img.width;
	vx_uint32 height = node->paramList[2]->u.img.height;
	if (node->paramList[2]->u.img.format != fmtIn)
		return VX_ERROR_INVALID_FORMAT;
	else if (!width || !height)
		return VX_ERROR_INVALID_DIMENSION;
	if (node->paramList[3]->u.thr.thresh_type != VX_THRESHOLD_TYPE_RANGE || 
		(node->paramList[3]->u.thr.data_type != VX_TYPE_UINT8 && node->paramList[3]->u.thr.data_type != VX_TYPE_UINT16 && node->paramList[3]->u.thr.data_type != VX_TYPE_INT16))
		return VX_ERROR_INVALID_TYPE;
	// set output info
	vx_meta_format meta;
	meta = &node->metaList[0];
	meta->data.u.img.width = width;
	meta->data.u.img.height = height;
	meta->data.u.img.format = VX_DF_IMAGE_U8;
	return VX_SUCCESS;
}
static int ValidateArguments_OpticalFlowPyrLK_XY_XY(AgoNode * node)
{
	AgoData * oldPyr = node->paramList[1];
	AgoData * newPyr = node->paramList[2];
	AgoData * oldXY = node->paramList[3];
	AgoData * newXYest = node->paramList[4];
	AgoData * termination = node->paramList[5];
	AgoData * epsilon = node->paramList[6];
	AgoData * num_iterations = node->paramList[7];
	AgoData * use_initial_estimate = node->paramList[8];
	if (oldXY->u.arr.itemtype != VX_TYPE_KEYPOINT || newXYest->u.arr.itemtype != VX_TYPE_KEYPOINT ||
		termination->u.scalar.type != VX_TYPE_ENUM || epsilon->u.scalar.type != VX_TYPE_FLOAT32 || 
		num_iterations->u.scalar.type != VX_TYPE_UINT32 || use_initial_estimate->u.scalar.type != VX_TYPE_BOOL)
		return VX_ERROR_INVALID_TYPE;
	else if (oldPyr->u.pyr.format != VX_DF_IMAGE_U8 || newPyr->u.pyr.format != VX_DF_IMAGE_U8)
		return VX_ERROR_INVALID_FORMAT;
	else if (!oldPyr->u.pyr.width || !oldPyr->u.pyr.height || !newPyr->u.pyr.width || !newPyr->u.pyr.height ||
		oldPyr->u.pyr.width != newPyr->u.pyr.width || oldPyr->u.pyr.height != newPyr->u.pyr.height ||
		!oldXY->u.arr.capacity || !newXYest->u.arr.capacity || oldXY->u.arr.capacity != newXYest->u.arr.capacity)
		return VX_ERROR_INVALID_DIMENSION;
	else if (termination->u.scalar.u.e != VX_TERM_CRITERIA_ITERATIONS && termination->u.scalar.u.e != VX_TERM_CRITERIA_EPSILON && termination->u.scalar.u.e != VX_TERM_CRITERIA_BOTH)
		return VX_ERROR_INVALID_VALUE;
	else if (oldPyr->u.pyr.scale != newPyr->u.pyr.scale || oldPyr->u.pyr.levels != newPyr->u.pyr.levels)
		return VX_ERROR_INVALID_VALUE;
	// set output info
	vx_meta_format meta;
	meta = &node->metaList[0];
	meta->data.u.arr.itemtype = VX_TYPE_KEYPOINT;
	meta->data.u.arr.capacity = oldXY->u.arr.capacity;
	return VX_SUCCESS;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// OpenVX 1.0 built-in kernels
//
int ovxKernel_Invalid(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: invalid kernel
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		// TBD: not implemented yet
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_ColorConvert(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_COLOR_CONVERT_* kernels
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		// TBD: not implemented yet
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		vx_df_image srcfmt = node->paramList[0]->u.img.format;
		if (srcfmt != VX_DF_IMAGE_RGB && srcfmt != VX_DF_IMAGE_RGBX && srcfmt != VX_DF_IMAGE_NV12 && srcfmt != VX_DF_IMAGE_NV21 &&
			srcfmt != VX_DF_IMAGE_IYUV && srcfmt != VX_DF_IMAGE_YUYV && srcfmt != VX_DF_IMAGE_UYVY)
			return VX_ERROR_INVALID_FORMAT;
		if (!width || !height || (width & 1) || (height & 1))
			return VX_ERROR_INVALID_DIMENSION;
		vx_df_image dstfmt = node->paramList[1]->u.img.format;
		if (dstfmt == VX_DF_IMAGE_VIRT)
			return VX_ERROR_INVALID_FORMAT;
		// set output image size is same as input image
		vx_meta_format meta = &node->metaList[1];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = dstfmt;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL
					| AGO_KERNEL_FLAG_DEVICE_GPU
#endif
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_ChannelExtract(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_CHANNEL_COPY_U8_U8 kernel for extracting from planar
	//       use VX_KERNEL_AMD_CHANNEL_EXTRACT_* kernels for extracting from interleaved
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
    else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		vx_df_image srcfmt = node->paramList[0]->u.img.format;
		if (srcfmt != VX_DF_IMAGE_RGB && srcfmt != VX_DF_IMAGE_RGBX && srcfmt != VX_DF_IMAGE_NV12 && srcfmt != VX_DF_IMAGE_NV21 &&
			srcfmt != VX_DF_IMAGE_IYUV && srcfmt != VX_DF_IMAGE_YUYV && srcfmt != VX_DF_IMAGE_UYVY && srcfmt != VX_DF_IMAGE_YUV4)
			return VX_ERROR_INVALID_FORMAT;
		if (!width || !height || (width & 1) || (height & 1))
			return VX_ERROR_INVALID_DIMENSION;
		vx_enum channel = node->paramList[1]->u.scalar.u.e;
		int channel_index = agoChannelEnum2Index(channel);
		int max_channel_index = (srcfmt == VX_DF_IMAGE_RGBX) ? 3 : 2;
		if (node->paramList[1]->u.scalar.type != VX_TYPE_ENUM || channel_index < 0 || channel_index > max_channel_index)
			return VX_ERROR_INVALID_VALUE;
		// set output image size is same as input image
		vx_meta_format meta = &node->metaList[2];
		vx_uint32 x_scale_factor_is_2 = 0, y_scale_factor_is_2 = 0;
		if (channel_index > 0) {
			if (node->paramList[0]->numChildren > 0) {
				x_scale_factor_is_2 = node->paramList[0]->children[1]->u.img.x_scale_factor_is_2;
				y_scale_factor_is_2 = node->paramList[0]->children[1]->u.img.y_scale_factor_is_2;
			}
			else if (srcfmt == VX_DF_IMAGE_YUYV || srcfmt == VX_DF_IMAGE_UYVY) {
				x_scale_factor_is_2 = 1;
			}
		}
		meta->data.u.img.width = width >> x_scale_factor_is_2;
		meta->data.u.img.height = height >> y_scale_factor_is_2;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_get_image_plane_nonusage) {
		status = VX_SUCCESS;
		if (node->funcExchange[0] == 0) {
			// mark that planes other than the specified channel are not used on input image
			vx_enum channel = node->paramList[1]->u.scalar.u.e;
			int channel_index = agoChannelEnum2Index(channel);
			for (vx_uint32 plane = 0; plane < node->paramList[0]->numChildren; plane++)
				node->funcExchange[1 + plane] = (plane != channel_index) ? 1 : 0;
		}
    }
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
					| AGO_KERNEL_FLAG_DEVICE_GPU
#endif				
					;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_ChannelCombine(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_CHANNEL_COPY_U8_U8 kernel for combining into planar
	//       use VX_KERNEL_AMD_CHANNEL_COMBINE_* kernels for combining into interleaved
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 chroma_x_scale_factor_is_2 = 0, chroma_y_scale_factor_is_2 = 0;
		vx_df_image dstfmt = node->paramList[4]->u.img.format;
		if (dstfmt == VX_DF_IMAGE_IYUV || dstfmt == VX_DF_IMAGE_NV12 || dstfmt == VX_DF_IMAGE_NV21)
			chroma_x_scale_factor_is_2 = chroma_y_scale_factor_is_2 = 1;
		else if (dstfmt == VX_DF_IMAGE_YUYV || dstfmt == VX_DF_IMAGE_UYVY)
			chroma_x_scale_factor_is_2 = 1;
		else if (dstfmt != VX_DF_IMAGE_RGB && dstfmt != VX_DF_IMAGE_RGBX && dstfmt != VX_DF_IMAGE_YUV4)
			return VX_ERROR_INVALID_FORMAT;
		vx_uint32 planeCount = 2;
		if (node->paramList[2]) planeCount++;
		else if (node->paramList[3]) planeCount++;
		if ((!node->paramList[2] && node->paramList[3]) || (planeCount != (node->paramList[4]->numChildren == 4 ? 4 : 3)))
			return VX_ERROR_INVALID_PARAMETERS;
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		for (vx_uint32 plane = 1; plane < planeCount; plane++) {
			if (node->paramList[plane]->u.img.format != VX_DF_IMAGE_U8)
				return VX_ERROR_INVALID_FORMAT;
			if (((node->paramList[plane]->u.img.width << chroma_x_scale_factor_is_2) != width) ||
				((node->paramList[plane]->u.img.height << chroma_y_scale_factor_is_2) != height))
				return VX_ERROR_INVALID_DIMENSION;
		}
		// set output image size is same as input image
		vx_meta_format meta = &node->metaList[4];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = dstfmt;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
					| AGO_KERNEL_FLAG_DEVICE_GPU
#endif				
					;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_Sobel3x3(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_SOBEL_* kernels
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		vx_df_image srcfmt = node->paramList[0]->u.img.format;
		if (srcfmt != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[1];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_S16;
		meta = &node->metaList[2];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_S16;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
					| AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_Magnitude(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_MAGNITUDE_S16_S16S16 kernel
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_S16 || node->paramList[1]->u.img.format != VX_DF_IMAGE_S16)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != node->paramList[1]->u.img.width || height != node->paramList[1]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[2];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_S16;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_Phase(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_PHASE_U8_S16S16 kernel
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_S16 || node->paramList[1]->u.img.format != VX_DF_IMAGE_S16)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != node->paramList[1]->u.img.width || height != node->paramList[1]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[2];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_ScaleImage(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_SCALE_IMAGE_* kernels
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8 || node->paramList[1]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!node->paramList[0]->u.img.width || !node->paramList[0]->u.img.height || !node->paramList[1]->u.img.width || !node->paramList[1]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (node->paramList[2]->u.scalar.type != VX_TYPE_ENUM)
			return VX_ERROR_INVALID_TYPE;
		else if (node->paramList[2]->u.scalar.u.e != VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR &&
			node->paramList[2]->u.scalar.u.e != VX_INTERPOLATION_TYPE_BILINEAR &&
			node->paramList[2]->u.scalar.u.e != VX_INTERPOLATION_TYPE_AREA)
			return VX_ERROR_INVALID_VALUE;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[1];
		meta->data.u.img.width = node->paramList[1]->u.img.width;
		meta->data.u.img.height = node->paramList[1]->u.img.height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_TableLookup(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_LUT_U8_U8 kernel
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8 || node->paramList[1]->u.lut.type != VX_TYPE_UINT8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[2];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                   
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_Histogram(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_HISTOGRAM_DATA_U8 kernel to get histogram of full/sub-image
	//       use VX_KERNEL_AMD_HISTOGRAM_MERGE_DATA_DATA kernel if sub-images are scheduled on multi-core
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_EqualizeHistogram(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_HISTOGRAM_DATA_U8 kernel to get histogram of full/sub-image
	//       use VX_KERNEL_AMD_EQUALIZE_DATA_DATA kernel to generate lut from histogram(s) for equalization
	//       use VX_KERNEL_AMD_LUT_U8_U8 kernels to equalize full/sub-images
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[1];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_AbsDiff(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_ABS_DIFF_U8_U8U8 kernel
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		vx_df_image format = node->paramList[0]->u.img.format;
		if ((format != VX_DF_IMAGE_U8 && format != VX_DF_IMAGE_S16) || node->paramList[1]->u.img.format != format)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != node->paramList[1]->u.img.width || height != node->paramList[1]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[2];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = format;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_MeanStdDev(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_MEAN_STD_DEV_DATA_U8 kernel to get sum and sum of squares on full/sub-images
	//       use VX_KERNEL_AMD_MEAN_STD_DEV_MERGE_DATA_DATA kernel to get mean and std-dev
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set scaler output types to FLOAT32
		vx_meta_format meta;
		meta = &node->metaList[1];
		meta->data.u.scalar.type = VX_TYPE_FLOAT32;
		meta = &node->metaList[2];
		meta->data.u.scalar.type = VX_TYPE_FLOAT32;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_Threshold(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_THRESHOLD_* kernels
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		if (node->paramList[1]->u.thr.data_type != VX_TYPE_UINT8)
			return VX_ERROR_INVALID_TYPE;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[2];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_IntegralImage(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_INTEGRAL_IMAGE_U32_U8 kernel
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[1];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U32;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_Dilate3x3(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_DILATE_* kernels
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[1];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_Erode3x3(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_ERODE_* kernels
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[1];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_Median3x3(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_MEDIAN_U8_U8_3x3 kernel
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[1];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_Box3x3(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_BOX_U8_U8_3x3 kernel
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[1];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_Gaussian3x3(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_GAUSSIAN_U8_U8_3x3 kernel
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[1];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_CustomConvolution(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_LINEAR_FILTER_* kernels
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (!node->paramList[1]->u.conv.columns || !node->paramList[1]->u.conv.rows)
			return VX_ERROR_INVALID_DIMENSION;
		vx_df_image_e dstfmt = VX_DF_IMAGE_S16;
		if (node->paramList[2]->u.img.format == VX_DF_IMAGE_U8)
			dstfmt = VX_DF_IMAGE_U8;
		// set output image sizes are same as input image size
		int M = (int) node->paramList[1]->u.conv.columns >> 1;
		int N = (int) node->paramList[1]->u.conv.rows >> 1;
		vx_meta_format meta;
		meta = &node->metaList[2];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = dstfmt;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_GaussianPyramid(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_SCALE_GAUSSIAN_* kernels recursively on each plane of the pyramid
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		AgoData * image = node->paramList[0];
		vx_uint32 width = image->u.img.width;
		vx_uint32 height = image->u.img.height;
		vx_enum format = image->u.img.format;
		vx_float32 scale = node->paramList[1]->u.pyr.scale;
		vx_size levels = node->paramList[1]->u.pyr.levels;
		if (format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (scale != VX_SCALE_PYRAMID_HALF && scale != VX_SCALE_PYRAMID_ORB)
			return VX_ERROR_INVALID_VALUE;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[1];
		meta->data.u.pyr.width = width;
		meta->data.u.pyr.height = height;
		meta->data.u.pyr.format = format;
		meta->data.u.pyr.levels = levels;
		meta->data.u.pyr.scale = scale;
		meta->data.u.pyr.rect_valid.start_x = image->u.img.rect_valid.start_x;
		meta->data.u.pyr.rect_valid.start_y = image->u.img.rect_valid.start_y;
		meta->data.u.pyr.rect_valid.end_x = image->u.img.rect_valid.end_x;
		meta->data.u.pyr.rect_valid.end_y = image->u.img.rect_valid.end_y;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_Accumulate(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_ADD_S16_S16U8_SAT kernel
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
    else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8 || node->paramList[1]->u.img.format != VX_DF_IMAGE_S16)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != node->paramList[1]->u.img.width || height != node->paramList[1]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_AccumulateWeighted(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_ACCUMULATE_WEIGHTED_U8_U8U8 kernel
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8 || node->paramList[2]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != node->paramList[2]->u.img.width || height != node->paramList[2]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (node->paramList[1]->u.scalar.type != VX_TYPE_FLOAT32)
			return VX_ERROR_INVALID_TYPE;
		else if (node->paramList[1]->u.scalar.u.f < 0.0f || node->paramList[1]->u.scalar.u.f > 1.0f)
			return VX_ERROR_INVALID_VALUE;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_AccumulateSquare(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_ADD_SQUARED_S16_S16U8 kernel
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
    else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8 || node->paramList[2]->u.img.format != VX_DF_IMAGE_S16)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != node->paramList[2]->u.img.width || height != node->paramList[2]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (node->paramList[1]->u.scalar.type != VX_TYPE_UINT32)
			return VX_ERROR_INVALID_TYPE;
		else if (node->paramList[1]->u.scalar.u.u > 15)
			return VX_ERROR_INVALID_VALUE;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_MinMaxLoc(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_MIN_MAX_LOC_DATA_U8/S16 kernels find min & max of full/sub-images
	//       use VX_KERNEL_AMD_MIN_MAX_LOC_MERGE_DATA_DATA if sub-images are used to find min & max
	//       use VX_KERNEL_AMD_MIN_MAX_LOC_MERGE_DATA_U8/S16DATA_* kernels find min & max of full/sub-images depending on configuration
	//       use VX_KERNEL_AMD_MIN_MAX_LOC_MERGE_DATA_DATA kernel if Loc/Count is used on sub-images
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8 && node->paramList[0]->u.img.format != VX_DF_IMAGE_S16)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output data info
		node->metaList[1].data.u.scalar.type = (node->paramList[0]->u.img.format == VX_DF_IMAGE_U8) ? VX_TYPE_UINT8 : VX_TYPE_INT16;
		node->metaList[2].data.u.scalar.type = (node->paramList[0]->u.img.format == VX_DF_IMAGE_U8) ? VX_TYPE_UINT8 : VX_TYPE_INT16;
		node->metaList[3].data.u.arr.itemtype = VX_TYPE_COORDINATES2D;
		node->metaList[3].data.u.arr.capacity = 0;
		node->metaList[4].data.u.arr.itemtype = VX_TYPE_COORDINATES2D;
		node->metaList[4].data.u.arr.capacity = 0;
		node->metaList[5].data.u.scalar.type = VX_TYPE_UINT32;
		node->metaList[6].data.u.scalar.type = VX_TYPE_UINT32;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_ConvertDepth(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_COLOR_DEPTH_* kernels
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8 && node->paramList[0]->u.img.format != VX_DF_IMAGE_S16)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (node->paramList[2]->u.scalar.type != VX_TYPE_ENUM || node->paramList[3]->u.scalar.type != VX_TYPE_INT32)
			return VX_ERROR_INVALID_TYPE;
		else if ((node->paramList[2]->u.scalar.u.e != VX_CONVERT_POLICY_WRAP && node->paramList[2]->u.scalar.u.e != VX_CONVERT_POLICY_SATURATE) ||
			     (node->paramList[3]->u.scalar.u.i < 0 || node->paramList[3]->u.scalar.u.i >= 8))
			return VX_ERROR_INVALID_VALUE;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[1];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = (node->paramList[0]->u.img.format == VX_DF_IMAGE_U8) ? VX_DF_IMAGE_S16 : VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_CannyEdgeDetector(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: Alternative#1: (1st level performance optimization)
	//         use VX_KERNEL_AMD_CANNY_SOBEL_U16_* or kernels to compute sobel magnitude
	//         use VX_KERNEL_AMD_CANNY_SUPP_THRESHOLD_U8XY_U16_3x3 kernel to threshold
	//         use VX_KERNEL_AMD_CANNY_EDGE_TRACE_U8_U8XY kernel to trace the edges
	//       Alternative#2: (2nd level performance optimization)
	//         use VX_KERNEL_AMD_CANNY_SOBEL_SUPP_THRESHOLD_U8XY_U8_* or kernels to compute sobel, non-max supression, and threshold
	//         use VX_KERNEL_AMD_CANNY_EDGE_TRACE_U8_U8XY kernel to trace the edges
	//       Alternative#3: (3rd level performance optimization)
	//         use VX_KERNEL_AMD_CANNY_SOBEL_U16_* or kernels to compute sobel magnitude
	//         use VX_KERNEL_AMD_CANNY_SUPP_THRESHOLD_U8_U16_3x3 kernel to threshold
	//         use VX_KERNEL_AMD_CANNY_EDGE_TRACE_U8_U8 kernel to trace the edges
	//       Alternative#4: (4th level performance optimization)
	//         use VX_KERNEL_AMD_CANNY_SOBEL_SUPP_THRESHOLD_U8_U8_* or kernels to compute sobel, non-max supression, and threshold
	//         use VX_KERNEL_AMD_CANNY_EDGE_TRACE_U8_U8 kernel to trace the edges
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		else if ((node->paramList[1]->u.thr.data_type != VX_TYPE_UINT8 && node->paramList[1]->u.thr.data_type != VX_TYPE_UINT16 && node->paramList[1]->u.thr.data_type != VX_TYPE_INT16) ||
			node->paramList[1]->u.thr.thresh_type != VX_THRESHOLD_TYPE_RANGE ||
			node->paramList[2]->u.scalar.type != VX_TYPE_INT32 || 
			node->paramList[3]->u.scalar.type != VX_TYPE_ENUM)
			return VX_ERROR_INVALID_TYPE;
		else if (node->paramList[3]->u.scalar.u.e != VX_NORM_L1 && node->paramList[3]->u.scalar.u.e != VX_NORM_L2)
			return VX_ERROR_INVALID_VALUE;
		// set output image sizes are same as input image size
		int N = node->paramList[2]->u.scalar.u.i >> 1;
		vx_meta_format meta;
		meta = &node->metaList[4];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_And(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_VX_KERNEL_AMD_AND_U8_U8U8 kernel
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8 || node->paramList[1]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != node->paramList[1]->u.img.width || height != node->paramList[1]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[2];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_Or(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_VX_KERNEL_AMD_OR_U8_U8U8 kernel
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8 || node->paramList[1]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != node->paramList[1]->u.img.width || height != node->paramList[1]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[2];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_Xor(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_VX_KERNEL_AMD_XOR_U8_U8U8 kernel
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8 || node->paramList[1]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != node->paramList[1]->u.img.width || height != node->paramList[1]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[2];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_Not(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_VX_KERNEL_AMD_NOT_U8_U8U8 kernel
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[1];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_Multiply(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_VX_KERNEL_AMD_MULTIPLY_* kernels
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if ((node->paramList[0]->u.img.format != VX_DF_IMAGE_U8  && node->paramList[0]->u.img.format != VX_DF_IMAGE_S16 && 
			 node->paramList[0]->u.img.format != VX_DF_IMAGE_RGB && node->paramList[0]->u.img.format != VX_DF_IMAGE_RGBX) ||
			(node->paramList[1]->u.img.format != VX_DF_IMAGE_U8  && node->paramList[1]->u.img.format != VX_DF_IMAGE_S16))
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != node->paramList[1]->u.img.width || height != node->paramList[1]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (node->paramList[2]->u.scalar.type != VX_TYPE_FLOAT32 || 
			node->paramList[3]->u.scalar.type != VX_TYPE_ENUM ||
			node->paramList[4]->u.scalar.type != VX_TYPE_ENUM)
			return VX_ERROR_INVALID_TYPE;
		else if ((node->paramList[3]->u.scalar.u.e != VX_CONVERT_POLICY_WRAP && node->paramList[3]->u.scalar.u.e != VX_CONVERT_POLICY_SATURATE) ||
			     (node->paramList[4]->u.scalar.u.e != VX_ROUND_POLICY_TO_ZERO && node->paramList[4]->u.scalar.u.e != VX_ROUND_POLICY_TO_NEAREST_EVEN))
			return VX_ERROR_INVALID_VALUE;
		// set output image sizes are same as input image size
		vx_df_image dstfmt = VX_DF_IMAGE_VIRT;
		if (node->paramList[0]->u.img.format == VX_DF_IMAGE_U8 && node->paramList[1]->u.img.format == VX_DF_IMAGE_U8)
			dstfmt = (node->paramList[5]->u.img.format == VX_DF_IMAGE_U8) ? VX_DF_IMAGE_U8 : VX_DF_IMAGE_S16;
		else if (node->paramList[0]->u.img.format == VX_DF_IMAGE_S16 || node->paramList[1]->u.img.format == VX_DF_IMAGE_S16)
			dstfmt = VX_DF_IMAGE_S16;
		else if (node->paramList[1]->u.img.format == VX_DF_IMAGE_U8 && (node->paramList[0]->u.img.format == VX_DF_IMAGE_RGB || node->paramList[0]->u.img.format == VX_DF_IMAGE_RGBX))
			dstfmt = node->paramList[0]->u.img.format;
		else
			return VX_ERROR_INVALID_FORMAT;
		vx_meta_format meta;
		meta = &node->metaList[5];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = dstfmt;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_Add(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_VX_KERNEL_AMD_ADD_* kernels
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if ((node->paramList[0]->u.img.format != VX_DF_IMAGE_U8 && node->paramList[0]->u.img.format != VX_DF_IMAGE_S16) ||
			(node->paramList[1]->u.img.format != VX_DF_IMAGE_U8 && node->paramList[1]->u.img.format != VX_DF_IMAGE_S16))
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != node->paramList[1]->u.img.width || height != node->paramList[1]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (node->paramList[2]->u.scalar.type != VX_TYPE_ENUM)
			return VX_ERROR_INVALID_TYPE;
		else if (node->paramList[2]->u.scalar.u.e != VX_CONVERT_POLICY_WRAP && node->paramList[2]->u.scalar.u.e != VX_CONVERT_POLICY_SATURATE)
			return VX_ERROR_INVALID_VALUE;
		// set output image sizes are same as input image size
		vx_df_image dstfmt = VX_DF_IMAGE_S16;
		if (node->paramList[0]->u.img.format == VX_DF_IMAGE_U8 &&
			node->paramList[1]->u.img.format == VX_DF_IMAGE_U8 &&
			node->paramList[3]->u.img.format == VX_DF_IMAGE_U8)
			dstfmt = VX_DF_IMAGE_U8;
		vx_meta_format meta;
		meta = &node->metaList[3];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = dstfmt;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_Subtract(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_VX_KERNEL_AMD_SUB_* kernels
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if ((node->paramList[0]->u.img.format != VX_DF_IMAGE_U8 && node->paramList[0]->u.img.format != VX_DF_IMAGE_S16) ||
			(node->paramList[1]->u.img.format != VX_DF_IMAGE_U8 && node->paramList[1]->u.img.format != VX_DF_IMAGE_S16))
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != node->paramList[1]->u.img.width || height != node->paramList[1]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (node->paramList[2]->u.scalar.type != VX_TYPE_ENUM)
			return VX_ERROR_INVALID_TYPE;
		else if (node->paramList[2]->u.scalar.u.e != VX_CONVERT_POLICY_WRAP && node->paramList[2]->u.scalar.u.e != VX_CONVERT_POLICY_SATURATE)
			return VX_ERROR_INVALID_VALUE;
		// set output image sizes are same as input image size
		vx_df_image dstfmt = VX_DF_IMAGE_S16;
		if (node->paramList[0]->u.img.format == VX_DF_IMAGE_U8 &&
			node->paramList[1]->u.img.format == VX_DF_IMAGE_U8 &&
			node->paramList[3]->u.img.format == VX_DF_IMAGE_U8)
			dstfmt = VX_DF_IMAGE_U8;
		vx_meta_format meta;
		meta = &node->metaList[3];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = dstfmt;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_WarpAffine(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_VX_KERNEL_AMD_WARP_AFFINE_* kernels
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (node->paramList[1]->u.mat.type != VX_TYPE_FLOAT32 || node->paramList[1]->u.mat.columns != 2 || node->paramList[1]->u.mat.rows != 3)
			return VX_ERROR_INVALID_FORMAT;
		else if (node->paramList[2]->u.scalar.type != VX_TYPE_ENUM)
			return VX_ERROR_INVALID_TYPE;
		else if (node->paramList[2]->u.scalar.u.e != VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR && node->paramList[2]->u.scalar.u.e != VX_INTERPOLATION_TYPE_BILINEAR)
			return VX_ERROR_INVALID_VALUE;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[3];
		meta->data.u.img.width = node->paramList[3]->u.img.width;
		meta->data.u.img.height = node->paramList[3]->u.img.height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_WarpPerspective(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_VX_KERNEL_AMD_WARP_PERSPECTIVE_* kernels
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (node->paramList[1]->u.mat.type != VX_TYPE_FLOAT32 || node->paramList[1]->u.mat.columns != 3 || node->paramList[1]->u.mat.rows != 3)
			return VX_ERROR_INVALID_FORMAT;
		else if (node->paramList[2]->u.scalar.type != VX_TYPE_ENUM)
			return VX_ERROR_INVALID_TYPE;
		else if (node->paramList[2]->u.scalar.u.e != VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR && node->paramList[2]->u.scalar.u.e != VX_INTERPOLATION_TYPE_BILINEAR)
			return VX_ERROR_INVALID_VALUE;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[3];
		meta->data.u.img.width = node->paramList[3]->u.img.width;
		meta->data.u.img.height = node->paramList[3]->u.img.height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_HarrisCorners(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_HARRIS_SOBEL_* kernels to compute Gx^2, Gx*Gy, Gy^2
	//       use VX_KERNEL_AMD_HARRIS_SCORE_* kernels to compute Vc
	//       use VX_KERNEL_AMD_HARRIS_MERGE_SORT_AND_PICK_XY_HVC kernel for final step
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (node->paramList[1]->u.scalar.type != VX_TYPE_FLOAT32 ||
			node->paramList[2]->u.scalar.type != VX_TYPE_FLOAT32 || node->paramList[3]->u.scalar.type != VX_TYPE_FLOAT32 ||
			node->paramList[4]->u.scalar.type != VX_TYPE_INT32 || node->paramList[5]->u.scalar.type != VX_TYPE_INT32)
			return VX_ERROR_INVALID_TYPE;
		else if (!(node->paramList[4]->u.scalar.u.i & 1) || node->paramList[4]->u.scalar.u.i < 3 || node->paramList[4]->u.scalar.u.i > 7 ||
				 !(node->paramList[5]->u.scalar.u.i & 1) || node->paramList[5]->u.scalar.u.i < 3 || node->paramList[5]->u.scalar.u.i > 7)
			return VX_ERROR_INVALID_VALUE;
		// set output data info
		node->metaList[6].data.u.arr.itemtype = VX_TYPE_KEYPOINT;
		node->metaList[6].data.u.arr.capacity = 0;
		node->metaList[7].data.u.scalar.type = VX_TYPE_SIZE;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_FastCorners(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_FAST_CORNERS_XY_U8_* kernels at full/sub-image level
	//       use VX_KERNEL_AMD_FAST_CORNER_MERGE_XY_XY kernel if sub-images are used
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (node->paramList[1]->u.scalar.type != VX_TYPE_FLOAT32 || node->paramList[2]->u.scalar.type != VX_TYPE_BOOL)
			return VX_ERROR_INVALID_TYPE;
		else if (node->paramList[2]->u.scalar.u.i < 0 || node->paramList[2]->u.scalar.u.i > 1)
			return VX_ERROR_INVALID_VALUE;
		// set output data info
		node->metaList[3].data.u.arr.itemtype = VX_TYPE_KEYPOINT;
		node->metaList[3].data.u.arr.capacity = 0;
		node->metaList[4].data.u.scalar.type = VX_TYPE_SIZE;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_OpticalFlowPyrLK(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_OPTICAL_FLOW_PYR_LK_XY_XY_* kernels
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.pyr.width;
		vx_uint32 height = node->paramList[0]->u.pyr.height;
		if (node->paramList[0]->u.pyr.format != VX_DF_IMAGE_U8 || node->paramList[1]->u.pyr.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != node->paramList[1]->u.pyr.width || height != node->paramList[1]->u.pyr.height ||
			node->paramList[0]->u.pyr.levels != node->paramList[1]->u.pyr.levels || node->paramList[0]->u.pyr.scale != node->paramList[1]->u.pyr.scale ||
			!node->paramList[2]->u.arr.capacity || node->paramList[2]->u.arr.capacity != node->paramList[3]->u.arr.capacity)
			return VX_ERROR_INVALID_DIMENSION;
		else if (node->paramList[2]->u.arr.itemtype != VX_TYPE_KEYPOINT || node->paramList[3]->u.arr.itemtype != VX_TYPE_KEYPOINT)
			return VX_ERROR_INVALID_FORMAT;
		else if (node->paramList[5]->u.scalar.type != VX_TYPE_ENUM || 
				  node->paramList[6]->u.scalar.type != VX_TYPE_FLOAT32 ||
				  node->paramList[7]->u.scalar.type != VX_TYPE_UINT32 ||
				  node->paramList[8]->u.scalar.type != VX_TYPE_BOOL ||
				  node->paramList[9]->u.scalar.type != VX_TYPE_SIZE)
			return VX_ERROR_INVALID_TYPE;
		else if ((node->paramList[5]->u.scalar.u.e != VX_TERM_CRITERIA_ITERATIONS &&
				  node->paramList[5]->u.scalar.u.e != VX_TERM_CRITERIA_EPSILON &&
				  node->paramList[5]->u.scalar.u.e != VX_TERM_CRITERIA_BOTH) ||
			     node->paramList[9]->u.scalar.u.s > AGO_OPTICALFLOWPYRLK_MAX_DIM)
			return VX_ERROR_INVALID_VALUE;
		// set output data info
		node->metaList[4].data.u.arr.itemtype = VX_TYPE_KEYPOINT;
		node->metaList[4].data.u.arr.capacity = node->paramList[2]->u.arr.capacity;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_Remap(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_REMAP_U8_U8_* kernels
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8 && node->paramList[0]->u.img.format != VX_DF_IMAGE_RGB && node->paramList[0]->u.img.format != VX_DF_IMAGE_RGBX)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != node->paramList[1]->u.remap.src_width || height != node->paramList[1]->u.remap.src_height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (node->paramList[2]->u.scalar.type != VX_TYPE_ENUM)
			return VX_ERROR_INVALID_TYPE;
		else if (node->paramList[2]->u.scalar.u.e != VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR && node->paramList[2]->u.scalar.u.e != VX_INTERPOLATION_TYPE_BILINEAR)
			return VX_ERROR_INVALID_VALUE;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[3];
		meta->data.u.img.width = node->paramList[1]->u.remap.dst_width;
		meta->data.u.img.height = node->paramList[1]->u.remap.dst_height;
		if (node->paramList[3]->u.img.format == VX_DF_IMAGE_VIRT || node->paramList[3]->u.img.format == node->paramList[0]->u.img.format)
			meta->data.u.img.format = node->paramList[0]->u.img.format;
		else if (node->paramList[3]->u.img.format == VX_DF_IMAGE_RGB && node->paramList[0]->u.img.format == VX_DF_IMAGE_RGBX)
			meta->data.u.img.format = node->paramList[3]->u.img.format;
		else
			return VX_ERROR_INVALID_FORMAT;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_HalfScaleGaussian(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_SCALE_GAUSSIAN_HALF_U8_U8_* kernels
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (node->paramList[2]->u.scalar.type != VX_TYPE_INT32)
			return VX_ERROR_INVALID_TYPE;
		else if (node->paramList[2]->u.scalar.u.i != 3 && node->paramList[2]->u.scalar.u.i != 5)
			return VX_ERROR_INVALID_VALUE;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		int N = node->paramList[2]->u.scalar.u.i >> 1;
		meta = &node->metaList[1];
		meta->data.u.img.width = (width + 1) >> 1;
		meta->data.u.img.height = (height + 1) >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int ovxKernel_Copy(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_COPY_* kernels
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		// TBD: not implemented yet
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		if (node->paramList[0]->ref.type != node->paramList[1]->ref.type)
			return VX_ERROR_INVALID_PARAMETERS;
		// set meta must be same as input
		vx_meta_format meta;
		meta = &node->metaList[1];
		meta->data.ref.type = node->paramList[0]->ref.type;
		memcpy(&meta->data.u, &node->paramList[0]->u, sizeof(meta->data.u));
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
					| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL
					| AGO_KERNEL_FLAG_DEVICE_GPU
#endif
					;
		status = VX_SUCCESS;
	}
	return status;
}

int ovxKernel_Select(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_SELECT_* kernels
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		// TBD: not implemented yet
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		if ((node->paramList[1]->ref.type != node->paramList[2]->ref.type) || (node->paramList[1]->ref.type != node->paramList[3]->ref.type))
			return VX_ERROR_INVALID_PARAMETERS;
		if (memcmp(&node->paramList[1]->u, &node->paramList[2]->u, sizeof(node->paramList[1]->u)) != 0)
			return VX_ERROR_INVALID_PARAMETERS;
		if (node->paramList[0]->u.scalar.type != VX_TYPE_BOOL)
			return VX_ERROR_INVALID_TYPE;
		// set meta must be same as input
		vx_meta_format meta;
		meta = &node->metaList[3];
		meta->data.ref.type = node->paramList[1]->ref.type;
		memcpy(&meta->data.u, &node->paramList[1]->u, sizeof(meta->data.u));
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = AGO_KERNEL_FLAG_SUBGRAPH
					| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL
					| AGO_KERNEL_FLAG_DEVICE_GPU
#endif
					;
		status = VX_SUCCESS;
	}
	return status;
}

#if ENABLE_OPENCL
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Local OpenCL Codegen Functions
//
static void agoCodeGenOpenCL_Threshold_U8_U8_Binary(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef Threshold_U8_U8_Binary_\n"
		"#define Threshold_U8_U8_Binary_\n"
		"void Threshold_U8_U8_Binary(U8x8 * p0, U8x8 p1, uint p2)\n"
		"{\n"
		"  U8x8 r;\n"
		"  float4 thr = (float4)amd_unpack0(p2);\n"
		"  r.s0 = amd_pack((amd_unpack(p1.s0) - thr) * (float4)256.0f);\n"
		"  r.s1 = amd_pack((amd_unpack(p1.s1) - thr) * (float4)256.0f);\n"
		"  *p0 = r;\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_Threshold_U8_U8_Range(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef Threshold_U8_U8_Range_\n"
		"#define Threshold_U8_U8_Range_\n"
		"void Threshold_U8_U8_Range(U8x8 * p0, U8x8 p1, uint2 p2)\n"
		"{\n"
		"  U8x8 r;\n"
		"  float4 thr0 = (float4)(amd_unpack0(p2.s0) - 1.0f);\n"
		"  float4 thr1 = (float4)(amd_unpack0(p2.s1) + 1.0f);\n"
		"  float4 pix0 = amd_unpack(p1.s0);\n"
		"  float4 pix1 = amd_unpack(p1.s1);\n"
		"  r.s0  = amd_pack((pix0 - thr0) * (float4)256.0f);\n"
		"  r.s0 &= amd_pack((thr1 - pix0) * (float4)256.0f);\n"
		"  r.s1  = amd_pack((pix1 - thr0) * (float4)256.0f);\n"
		"  r.s1 &= amd_pack((thr1 - pix1) * (float4)256.0f);\n"
		"  *p0 = r;\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_Add_S16_S16U8_Sat(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef Add_S16_S16U8_Sat_\n"
		"#define Add_S16_S16U8_Sat_\n"
		"void Add_S16_S16U8_Sat(S16x8 * p0, S16x8 p1, U8x8 p2)\n"
		"{\n"
		"  S16x8 r;\n"
		"  r.s0  = (int)(clamp((float)(((int)(p1.s0) << 16) >> 16) + amd_unpack0(p2.s0), -32768.0f, 32767.0f)) & 0x0000ffff;\n"
		"  r.s0 |= (int)(clamp((float)( (int)(p1.s0)        >> 16) + amd_unpack1(p2.s0), -32768.0f, 32767.0f)) << 16;\n"
		"  r.s1  = (int)(clamp((float)(((int)(p1.s1) << 16) >> 16) + amd_unpack2(p2.s0), -32768.0f, 32767.0f)) & 0x0000ffff;\n"
		"  r.s1 |= (int)(clamp((float)( (int)(p1.s1)        >> 16) + amd_unpack3(p2.s0), -32768.0f, 32767.0f)) << 16;\n"
		"  r.s2  = (int)(clamp((float)(((int)(p1.s2) << 16) >> 16) + amd_unpack0(p2.s1), -32768.0f, 32767.0f)) & 0x0000ffff;\n"
		"  r.s2 |= (int)(clamp((float)( (int)(p1.s2)        >> 16) + amd_unpack1(p2.s1), -32768.0f, 32767.0f)) << 16;\n"
		"  r.s3  = (int)(clamp((float)(((int)(p1.s3) << 16) >> 16) + amd_unpack2(p2.s1), -32768.0f, 32767.0f)) & 0x0000ffff;\n"
		"  r.s3 |= (int)(clamp((float)( (int)(p1.s3)        >> 16) + amd_unpack3(p2.s1), -32768.0f, 32767.0f)) << 16;\n"
		"  *p0 = r;\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_AbsDiff_S16_S16S16_Sat(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef AbsDiff_S16_S16S16_Sat_\n"
		"#define AbsDiff_S16_S16S16_Sat_\n"
		"void AbsDiff_S16_S16S16_Sat(S16x8 * p0, S16x8 p1, S16x8 p2)\n"
		"{\n"
		"  S16x8 r;\n"
		"  r.s0  = min(abs_diff((((int)(p1.s0) << 16) >> 16), (((int)(p2.s0) << 16) >> 16)), 32767u);\n"
		"  r.s0 |= min(abs_diff(( (int)(p1.s0)        >> 16), ( (int)(p2.s0)        >> 16)), 32767u) << 16;\n"
		"  r.s1  = min(abs_diff((((int)(p1.s1) << 16) >> 16), (((int)(p2.s1) << 16) >> 16)), 32767u);\n"
		"  r.s1 |= min(abs_diff(( (int)(p1.s1)        >> 16), ( (int)(p2.s1)        >> 16)), 32767u) << 16;\n"
		"  r.s2  = min(abs_diff((((int)(p1.s2) << 16) >> 16), (((int)(p2.s2) << 16) >> 16)), 32767u);\n"
		"  r.s2 |= min(abs_diff(( (int)(p1.s2)        >> 16), ( (int)(p2.s2)        >> 16)), 32767u) << 16;\n"
		"  r.s3  = min(abs_diff((((int)(p1.s3) << 16) >> 16), (((int)(p2.s3) << 16) >> 16)), 32767u);\n"
		"  r.s3 |= min(abs_diff(( (int)(p1.s3)        >> 16), ( (int)(p2.s3)        >> 16)), 32767u) << 16;\n"
		"  *p0 = r;\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_ChannelExtract_U8_U24_Pos0(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef ChannelExtract_U8_U24_Pos0_\n"
		"#define ChannelExtract_U8_U24_Pos0_\n"
		"void ChannelExtract_U8_U24_Pos0(U8x8 * p0, U24x8 p1)\n"
		"{\n"
		"  U8x8 r;\n"
		"  r.s0 = amd_pack((float4)(amd_unpack0(p1.s0), amd_unpack3(p1.s0), amd_unpack2(p1.s1), amd_unpack1(p1.s2)));\n"
		"  r.s1 = amd_pack((float4)(amd_unpack0(p1.s3), amd_unpack3(p1.s3), amd_unpack2(p1.s4), amd_unpack1(p1.s5)));\n"
		"  *p0 = r;\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_ChannelExtract_U8_U24_Pos1(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef ChannelExtract_U8_U24_Pos1_\n"
		"#define ChannelExtract_U8_U24_Pos1_\n"
		"void ChannelExtract_U8_U24_Pos1(U8x8 * p0, U24x8 p1)\n"
		"{\n"
		"  U8x8 r;\n"
		"  r.s0 = amd_pack((float4)(amd_unpack1(p1.s0), amd_unpack0(p1.s1), amd_unpack3(p1.s1), amd_unpack2(p1.s2)));\n"
		"  r.s1 = amd_pack((float4)(amd_unpack1(p1.s3), amd_unpack0(p1.s4), amd_unpack3(p1.s4), amd_unpack2(p1.s5)));\n"
		"  *p0 = r;\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_ChannelExtract_U8_U24_Pos2(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef ChannelExtract_U8_U24_Pos2_\n"
		"#define ChannelExtract_U8_U24_Pos2_\n"
		"void ChannelExtract_U8_U24_Pos2(U8x8 * p0, U24x8 p1)\n"
		"{\n"
		"  U8x8 r;\n"
		"  r.s0 = amd_pack((float4)(amd_unpack2(p1.s0), amd_unpack1(p1.s1), amd_unpack0(p1.s2), amd_unpack3(p1.s2)));\n"
		"  r.s1 = amd_pack((float4)(amd_unpack2(p1.s3), amd_unpack1(p1.s4), amd_unpack0(p1.s5), amd_unpack3(p1.s5)));\n"
		"  *p0 = r;\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_ChannelExtract_U8_U32_Pos0(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef ChannelExtract_U8_U32_Pos0_\n"
		"#define ChannelExtract_U8_U32_Pos0_\n"
		"void ChannelExtract_U8_U32_Pos0(U8x8 * p0, U32x8 p1)\n"
		"{\n"
		"  U8x8 r;\n"
		"  r.s0 = amd_pack((float4)(amd_unpack0(p1.s0), amd_unpack0(p1.s1), amd_unpack0(p1.s2), amd_unpack0(p1.s3)));\n"
		"  r.s1 = amd_pack((float4)(amd_unpack0(p1.s4), amd_unpack0(p1.s5), amd_unpack0(p1.s6), amd_unpack0(p1.s7)));\n"
		"  *p0 = r;\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_ChannelExtract_U8_U32_Pos1(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef ChannelExtract_U8_U32_Pos1_\n"
		"#define ChannelExtract_U8_U32_Pos1_\n"
		"void ChannelExtract_U8_U32_Pos1(U8x8 * p0, U32x8 p1)\n"
		"{\n"
		"  U8x8 r;\n"
		"  r.s0 = amd_pack((float4)(amd_unpack1(p1.s0), amd_unpack1(p1.s1), amd_unpack1(p1.s2), amd_unpack1(p1.s3)));\n"
		"  r.s1 = amd_pack((float4)(amd_unpack1(p1.s4), amd_unpack1(p1.s5), amd_unpack1(p1.s6), amd_unpack1(p1.s7)));\n"
		"  *p0 = r;\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_ChannelExtract_U8_U32_Pos2(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef ChannelExtract_U8_U32_Pos2_\n"
		"#define ChannelExtract_U8_U32_Pos2_\n"
		"void ChannelExtract_U8_U32_Pos2(U8x8 * p0, U32x8 p1)\n"
		"{\n"
		"  U8x8 r;\n"
		"  r.s0 = amd_pack((float4)(amd_unpack2(p1.s0), amd_unpack2(p1.s1), amd_unpack2(p1.s2), amd_unpack2(p1.s3)));\n"
		"  r.s1 = amd_pack((float4)(amd_unpack2(p1.s4), amd_unpack2(p1.s5), amd_unpack2(p1.s6), amd_unpack2(p1.s7)));\n"
		"  *p0 = r;\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_ChannelExtract_U8_U32_Pos3(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef ChannelExtract_U8_U32_Pos3_\n"
		"#define ChannelExtract_U8_U32_Pos3_\n"
		"void ChannelExtract_U8_U32_Pos3(U8x8 * p0, U32x8 p1)\n"
		"{\n"
		"  U8x8 r;\n"
		"  r.s0 = amd_pack((float4)(amd_unpack3(p1.s0), amd_unpack3(p1.s1), amd_unpack3(p1.s2), amd_unpack3(p1.s3)));\n"
		"  r.s1 = amd_pack((float4)(amd_unpack3(p1.s4), amd_unpack3(p1.s5), amd_unpack3(p1.s6), amd_unpack3(p1.s7)));\n"
		"  *p0 = r;\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_ColorConvert_Y_RGB(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef ColorConvert_Y_RGB_\n"
		"#define ColorConvert_Y_RGB_\n"
		"void ColorConvert_Y_RGB(U8x8 * p0, U24x8 p1)\n"
		"{\n"
		"  U8x8 r;\n"
		"  float4 f;\n"
		"  float3 cY = (float3)(0.2126f, 0.7152f, 0.0722f);\n"
		"  f.s0 = dot(cY, (float3)(amd_unpack0(p1.s0), amd_unpack1(p1.s0), amd_unpack2(p1.s0))); \n"
		"  f.s1 = dot(cY, (float3)(amd_unpack3(p1.s0), amd_unpack0(p1.s1), amd_unpack1(p1.s1))); \n"
		"  f.s2 = dot(cY, (float3)(amd_unpack2(p1.s1), amd_unpack3(p1.s1), amd_unpack0(p1.s2))); \n"
		"  f.s3 = dot(cY, (float3)(amd_unpack1(p1.s2), amd_unpack2(p1.s2), amd_unpack3(p1.s2))); \n"
		"  r.s0 = amd_pack(f);\n"
		"  f.s0 = dot(cY, (float3)(amd_unpack0(p1.s3), amd_unpack1(p1.s3), amd_unpack2(p1.s3))); \n"
		"  f.s1 = dot(cY, (float3)(amd_unpack3(p1.s3), amd_unpack0(p1.s4), amd_unpack1(p1.s4))); \n"
		"  f.s2 = dot(cY, (float3)(amd_unpack2(p1.s4), amd_unpack3(p1.s4), amd_unpack0(p1.s5))); \n"
		"  f.s3 = dot(cY, (float3)(amd_unpack1(p1.s5), amd_unpack2(p1.s5), amd_unpack3(p1.s5))); \n"
		"  r.s1 = amd_pack(f);\n"
		"  *p0 = r;\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_ColorConvert_U_RGB(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef ColorConvert_U_RGB_\n"
		"#define ColorConvert_U_RGB_\n"
		"void ColorConvert_U_RGB(U8x8 * p0, U24x8 p1)\n"
		"{\n"
		"  U8x8 r;\n"
		"  float4 f;\n"
		"  float3 cU = (float3)(-0.1146f, -0.3854f, 0.5f);\n"
		"  f.s0 = dot(cU, (float3)(amd_unpack0(p1.s0), amd_unpack1(p1.s0), amd_unpack2(p1.s0))) + 128.0f; \n"
		"  f.s1 = dot(cU, (float3)(amd_unpack3(p1.s0), amd_unpack0(p1.s1), amd_unpack1(p1.s1))) + 128.0f; \n"
		"  f.s2 = dot(cU, (float3)(amd_unpack2(p1.s1), amd_unpack3(p1.s1), amd_unpack0(p1.s2))) + 128.0f; \n"
		"  f.s3 = dot(cU, (float3)(amd_unpack1(p1.s2), amd_unpack2(p1.s2), amd_unpack3(p1.s2))) + 128.0f; \n"
		"  r.s0 = amd_pack(f);\n"
		"  f.s0 = dot(cU, (float3)(amd_unpack0(p1.s3), amd_unpack1(p1.s3), amd_unpack2(p1.s3))) + 128.0f; \n"
		"  f.s1 = dot(cU, (float3)(amd_unpack3(p1.s3), amd_unpack0(p1.s4), amd_unpack1(p1.s4))) + 128.0f; \n"
		"  f.s2 = dot(cU, (float3)(amd_unpack2(p1.s4), amd_unpack3(p1.s4), amd_unpack0(p1.s5))) + 128.0f; \n"
		"  f.s3 = dot(cU, (float3)(amd_unpack1(p1.s5), amd_unpack2(p1.s5), amd_unpack3(p1.s5))) + 128.0f; \n"
		"  r.s1 = amd_pack(f);\n"
		"  *p0 = r;\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_ColorConvert_V_RGB(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef ColorConvert_V_RGB_\n"
		"#define ColorConvert_V_RGB_\n"
		"void ColorConvert_V_RGB(U8x8 * p0, U24x8 p1)\n"
		"{\n"
		"  U8x8 r;\n"
		"  float4 f;\n"
		"  float3 cV = (float3)(0.5f, -0.4542f, -0.0458f);\n"
		"  f.s0 = dot(cV, (float3)(amd_unpack0(p1.s0), amd_unpack1(p1.s0), amd_unpack2(p1.s0))) + 128.0f; \n"
		"  f.s1 = dot(cV, (float3)(amd_unpack3(p1.s0), amd_unpack0(p1.s1), amd_unpack1(p1.s1))) + 128.0f; \n"
		"  f.s2 = dot(cV, (float3)(amd_unpack2(p1.s1), amd_unpack3(p1.s1), amd_unpack0(p1.s2))) + 128.0f; \n"
		"  f.s3 = dot(cV, (float3)(amd_unpack1(p1.s2), amd_unpack2(p1.s2), amd_unpack3(p1.s2))) + 128.0f; \n"
		"  r.s0 = amd_pack(f);\n"
		"  f.s0 = dot(cV, (float3)(amd_unpack0(p1.s3), amd_unpack1(p1.s3), amd_unpack2(p1.s3))) + 128.0f; \n"
		"  f.s1 = dot(cV, (float3)(amd_unpack3(p1.s3), amd_unpack0(p1.s4), amd_unpack1(p1.s4))) + 128.0f; \n"
		"  f.s2 = dot(cV, (float3)(amd_unpack2(p1.s4), amd_unpack3(p1.s4), amd_unpack0(p1.s5))) + 128.0f; \n"
		"  f.s3 = dot(cV, (float3)(amd_unpack1(p1.s5), amd_unpack2(p1.s5), amd_unpack3(p1.s5))) + 128.0f; \n"
		"  r.s1 = amd_pack(f);\n"
		"  *p0 = r;\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_ColorConvert_Y_RGBX(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef ColorConvert_Y_RGBX_\n"
		"#define ColorConvert_Y_RGBX_\n"
		"void ColorConvert_Y_RGBX(U8x8 * p0, U32x8 p1)\n"
		"{\n"
		"  U8x8 r;\n"
		"  float4 f;\n"
		"  float3 cY = (float3)(0.2126f, 0.7152f, 0.0722f);\n"
		"  f.s0 = dot(cY, (float3)(amd_unpack0(p1.s0), amd_unpack1(p1.s0), amd_unpack2(p1.s0))); \n"
		"  f.s1 = dot(cY, (float3)(amd_unpack0(p1.s1), amd_unpack1(p1.s1), amd_unpack2(p1.s1))); \n"
		"  f.s2 = dot(cY, (float3)(amd_unpack0(p1.s2), amd_unpack1(p1.s2), amd_unpack2(p1.s2))); \n"
		"  f.s3 = dot(cY, (float3)(amd_unpack0(p1.s3), amd_unpack1(p1.s3), amd_unpack2(p1.s3))); \n"
		"  r.s0 = amd_pack(f);\n"
		"  f.s0 = dot(cY, (float3)(amd_unpack0(p1.s4), amd_unpack1(p1.s4), amd_unpack2(p1.s4))); \n"
		"  f.s1 = dot(cY, (float3)(amd_unpack0(p1.s5), amd_unpack1(p1.s5), amd_unpack2(p1.s5))); \n"
		"  f.s2 = dot(cY, (float3)(amd_unpack0(p1.s6), amd_unpack1(p1.s6), amd_unpack2(p1.s6))); \n"
		"  f.s3 = dot(cY, (float3)(amd_unpack0(p1.s7), amd_unpack1(p1.s7), amd_unpack2(p1.s7))); \n"
		"  r.s1 = amd_pack(f);\n"
		"  *p0 = r;\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_ColorConvert_U_RGBX(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef ColorConvert_U_RGBX_\n"
		"#define ColorConvert_U_RGBX_\n"
		"void ColorConvert_U_RGBX(U8x8 * p0, U32x8 p1)\n"
		"{\n"
		"  U8x8 r;\n"
		"  float4 f;\n"
		"  float3 cU = (float3)(-0.1146f, -0.3854f, 0.5f);\n"
		"  f.s0 = dot(cU, (float3)(amd_unpack0(p1.s0), amd_unpack1(p1.s0), amd_unpack2(p1.s0))) + 128.0f; \n"
		"  f.s1 = dot(cU, (float3)(amd_unpack0(p1.s1), amd_unpack1(p1.s1), amd_unpack2(p1.s1))) + 128.0f; \n"
		"  f.s2 = dot(cU, (float3)(amd_unpack0(p1.s2), amd_unpack1(p1.s2), amd_unpack2(p1.s2))) + 128.0f; \n"
		"  f.s3 = dot(cU, (float3)(amd_unpack0(p1.s3), amd_unpack1(p1.s3), amd_unpack2(p1.s3))) + 128.0f; \n"
		"  r.s0 = amd_pack(f);\n"
		"  f.s0 = dot(cU, (float3)(amd_unpack0(p1.s4), amd_unpack1(p1.s4), amd_unpack2(p1.s4))) + 128.0f; \n"
		"  f.s1 = dot(cU, (float3)(amd_unpack0(p1.s5), amd_unpack1(p1.s5), amd_unpack2(p1.s5))) + 128.0f; \n"
		"  f.s2 = dot(cU, (float3)(amd_unpack0(p1.s6), amd_unpack1(p1.s6), amd_unpack2(p1.s6))) + 128.0f; \n"
		"  f.s3 = dot(cU, (float3)(amd_unpack0(p1.s7), amd_unpack1(p1.s7), amd_unpack2(p1.s7))) + 128.0f; \n"
		"  r.s1 = amd_pack(f);\n"
		"  *p0 = r;\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_ColorConvert_V_RGBX(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef ColorConvert_V_RGBX_\n"
		"#define ColorConvert_V_RGBX_\n"
		"void ColorConvert_V_RGBX(U8x8 * p0, U32x8 p1)\n"
		"{\n"
		"  U8x8 r;\n"
		"  float4 f;\n"
		"  float3 cV = (float3)(0.5f, -0.4542f, -0.0458f);\n"
		"  f.s0 = dot(cV, (float3)(amd_unpack0(p1.s0), amd_unpack1(p1.s0), amd_unpack2(p1.s0))) + 128.0f; \n"
		"  f.s1 = dot(cV, (float3)(amd_unpack0(p1.s1), amd_unpack1(p1.s1), amd_unpack2(p1.s1))) + 128.0f; \n"
		"  f.s2 = dot(cV, (float3)(amd_unpack0(p1.s2), amd_unpack1(p1.s2), amd_unpack2(p1.s2))) + 128.0f; \n"
		"  f.s3 = dot(cV, (float3)(amd_unpack0(p1.s3), amd_unpack1(p1.s3), amd_unpack2(p1.s3))) + 128.0f; \n"
		"  r.s0 = amd_pack(f);\n"
		"  f.s0 = dot(cV, (float3)(amd_unpack0(p1.s4), amd_unpack1(p1.s4), amd_unpack2(p1.s4))) + 128.0f; \n"
		"  f.s1 = dot(cV, (float3)(amd_unpack0(p1.s5), amd_unpack1(p1.s5), amd_unpack2(p1.s5))) + 128.0f; \n"
		"  f.s2 = dot(cV, (float3)(amd_unpack0(p1.s6), amd_unpack1(p1.s6), amd_unpack2(p1.s6))) + 128.0f; \n"
		"  f.s3 = dot(cV, (float3)(amd_unpack0(p1.s7), amd_unpack1(p1.s7), amd_unpack2(p1.s7))) + 128.0f; \n"
		"  r.s1 = amd_pack(f);\n"
		"  *p0 = r;\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_BilinearSample(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef BilinearSample_\n"
		"#define BilinearSample_\n"
		"float BilinearSample(__global uchar *p, uint ystride, uint xstride, float fy0, float fy1, int x, float fx0, float fx1)\n"
		"{\n"
		"  float4 f;\n"
		"  p += x;\n"
		"  f.s0 = amd_unpack0((uint)p[0]);\n"
		"  f.s1 = amd_unpack0((uint)p[xstride]);\n"
		"  p += ystride;\n"
		"  f.s2 = amd_unpack0((uint)p[0]);\n"
		"  f.s3 = amd_unpack0((uint)p[xstride]);\n"
		"  f.s0 = mad(f.s0, fx0, f.s1 * fx1);\n"
		"  f.s2 = mad(f.s2, fx0, f.s3 * fx1);\n"
		"  f.s0 = mad(f.s0, fy0, f.s2 * fy1);\n"
		"  return f.s0;\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_BilinearSampleFXY(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef BilinearSampleFXY_\n"
		"#define BilinearSampleFXY_\n"
		"float BilinearSampleFXY(__global uchar *p, uint stride, float sx, float sy)\n"
		"{\n"
		"  float fx0, fx1, fy0, fy1, ii; uint x, y;\n"
		"  fx1 = fract(sx, &ii); fx0 = 1.0f - fx1; x = (uint)ii;\n"
		"  fy1 = fract(sy, &ii); fy0 = 1.0f - fy1; y = (uint)ii;\n"
		"  p += mad24(stride, y, x);\n"
		"  return BilinearSample(p, stride, 1, fy0, fy1, 0, fx0, fx1);\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_BilinearSampleFXYConstantForRemap(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef BilinearSampleFXYConstantForRemap_\n"
		"#define BilinearSampleFXYConstantForRemap_\n"
		"float BilinearSampleFXYConstantForRemap(__global uchar *p, uint stride, uint width, uint height, float sx, float sy, uint borderValue)\n"
		"{\n"
		"  float fx0, fx1, fy0, fy1, ii; int x, y;\n"
		"  fx1 = fract(sx, &ii); fx0 = 1.0f - fx1; x = (int)floor(sx);\n"
		"  fy1 = fract(sy, &ii); fy0 = 1.0f - fy1; y = (int)floor(sy);\n"
		"  if (((uint)x) < width - 1 && ((uint)y) < height - 1) {\n"
		"  	p += y*stride;\n"
		"  	return BilinearSample(p, stride, 1, fy0, fy1, x, fx0, fx1);\n"
		"  }\n"
		"  else {\n"
		"  	return amd_unpack0(borderValue);\n"
		"  }\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_SampleWithConstBorder(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef SampleWithConstBorder_\n"
		"#define SampleWithConstBorder_\n"
		"uint SampleWithConstBorder(__global uchar *p, int x, int y, uint width, uint height, uint stride, uint borderValue)\n"
		"{\n"
		"  uint pixelValue = borderValue;\n"
		"  if (x >= 0 && y >= 0 && x < width && y < height) {\n"
		"  	pixelValue = p[y*stride + x];\n"
		"  }\n"
		"  return pixelValue;\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_BilinearSampleWithConstBorder(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef BilinearSampleWithConstBorder_\n"
		"#define BilinearSampleWithConstBorder_\n"
		"float BilinearSampleWithConstBorder(__global uchar *p, int x, int y, uint width, uint height, uint stride, float fx0, float fx1, float fy0, float fy1, uint borderValue)\n"
		"{\n"
		"  float4 f;\n"
		"  f.s0 = amd_unpack0(SampleWithConstBorder(p, x, y, width, height, stride, borderValue));\n"
		"  f.s1 = amd_unpack0(SampleWithConstBorder(p, x + 1, y, width, height, stride, borderValue));\n"
		"  f.s2 = amd_unpack0(SampleWithConstBorder(p, x, y + 1, width, height, stride, borderValue));\n"
		"  f.s3 = amd_unpack0(SampleWithConstBorder(p, x + 1, y + 1, width, height, stride, borderValue));\n"
		"  f.s0 = mad(f.s0, fx0, f.s1 * fx1);\n"
		"  f.s2 = mad(f.s2, fx0, f.s3 * fx1);\n"
		"  f.s0 = mad(f.s0, fy0, f.s2 * fy1);\n"
		"  return f.s0;\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_BilinearSampleFXYConstant(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef BilinearSampleFXYConstant_\n"
		"#define BilinearSampleFXYConstant_\n"
		"float BilinearSampleFXYConstant(__global uchar *p, uint stride, uint width, uint height, float sx, float sy, uint borderValue)\n"
		"{\n"
		"  float fx0, fx1, fy0, fy1, ii; int x, y;\n"
		"  fx1 = fract(sx, &ii); fx0 = 1.0f - fx1; x = (int)ii;\n"
		"  fy1 = fract(sy, &ii); fy0 = 1.0f - fy1; y = (int)ii;\n"
		"  if (((uint)x) < width && ((uint)y) < height) {\n"
		"  	p += y*stride;\n"
		"  	return BilinearSample(p, stride, 1, fy0, fy1, x, fx0, fx1);\n"
		"  }\n"
		"  else {\n"
		"  	return BilinearSampleWithConstBorder(p, x, y, width, height, stride, fx0, fx1, fy0, fy1, borderValue);\n"
		"  }\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_ClampPixelCoordinatesToBorder(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef ClampPixelCoordinatesToBorder_\n"
		"#define ClampPixelCoordinatesToBorder_\n"
		"uint2 ClampPixelCoordinatesToBorder(float f, uint limit, uint stride)\n"
		"{\n"
		"  uint2 vstride;\n"
		"  vstride.s0 = select((uint)f, 0u, f < 0);\n"
		"  vstride.s1 = select(stride, 0u, f < 0);\n"
		"  vstride.s0 = select(vstride.s0, limit, f >= limit);\n"
		"  vstride.s1 = select(vstride.s1, 0u, f >= limit);\n"
		"  return vstride;\n"
		"}\n"
		"#endif\n"
		);
}
static void agoCodeGenOpenCL_ScaleImage_U8_U8_Bilinear(std::string& opencl_code)
{
	opencl_code += OPENCL_FORMAT(
		"#ifndef ScaleImage_U8_U8_Bilinear_\n"
		"#define ScaleImage_U8_U8_Bilinear_\n"
		"void ScaleImage_U8_U8_Bilinear(U8x8 * r, uint x, uint y, __global uchar * p, uint stride, float4 scaleInfo)\n"
		"{\n"
		"  U8x8 rv;\n"
		"  float fx, fy, fint, frac, fy0, fy1;\n"
		"  float4 f;\n"
		"  fy = mad((float)y, scaleInfo.s1, scaleInfo.s3);\n"
		"  fy0 = floor(fy); fy1 = fy - fy0; fy0 = 1.0f - fy1;\n"
		"  p += mul24((uint)fy, stride);\n"
		"  fx = mad((float)x, scaleInfo.s0, scaleInfo.s2); fint = floor(fx); frac = fx - fint; f.s0 = BilinearSample(p, stride, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
		"  fx += scaleInfo.s0;                             fint = floor(fx); frac = fx - fint; f.s1 = BilinearSample(p, stride, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
		"  fx += scaleInfo.s0;                             fint = floor(fx); frac = fx - fint; f.s2 = BilinearSample(p, stride, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
		"  fx += scaleInfo.s0;                             fint = floor(fx); frac = fx - fint; f.s3 = BilinearSample(p, stride, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
		"  rv.s0 = amd_pack(f);\n"
		"  fx += scaleInfo.s0;                             fint = floor(fx); frac = fx - fint; f.s0 = BilinearSample(p, stride, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
		"  fx += scaleInfo.s0;                             fint = floor(fx); frac = fx - fint; f.s1 = BilinearSample(p, stride, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
		"  fx += scaleInfo.s0;                             fint = floor(fx); frac = fx - fint; f.s2 = BilinearSample(p, stride, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
		"  fx += scaleInfo.s0;                             fint = floor(fx); frac = fx - fint; f.s3 = BilinearSample(p, stride, 1, fy0, fy1, (int)fint, 1.0f - frac, frac);\n"
		"  rv.s1 = amd_pack(f);\n"
		"  *r = rv;\n"
		"}\n"
		"#endif\n"
		);
}
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// AMD low-level kernels
//
int agoKernel_Set00_U8(AgoNode * node, AgoKernelCommand cmd)
{
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		if (HafCpu_MemSet_U8(node->paramList[0]->size, node->paramList[0]->buffer, 0x00)) {
			status = VX_FAILURE;
		}
    }
    else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT(node, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
    }
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s(U8x8 * p0)\n"
			"{\n"
			"  *p0 = (U8x8)(0);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL
					| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		out->u.img.rect_valid.start_x = 0;
		out->u.img.rect_valid.start_y = 0;
		out->u.img.rect_valid.end_x = out->u.img.width;
		out->u.img.rect_valid.end_y = out->u.img.height;
	}
	return status;
}

int agoKernel_SetFF_U8(AgoNode * node, AgoKernelCommand cmd)
{
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		if (HafCpu_MemSet_U8(node->paramList[0]->size, node->paramList[0]->buffer, 0xFF)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT(node, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0)\n"
			"{\n"
			"  *p0 = (U8x8)(0xffffffff);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL
					| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		out->u.img.rect_valid.start_x = 0;
		out->u.img.rect_valid.start_y = 0;
		out->u.img.rect_valid.end_x = out->u.img.width;
		out->u.img.rect_valid.end_y = out->u.img.height;
	}
	return status;
}

int agoKernel_Not_U8_U8(AgoNode * node, AgoKernelCommand cmd)
{
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_Not_U8_U8(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1)\n"
			"{\n"
			"  *p0 = ~p1;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_Not_U8_U1(AgoNode * node, AgoKernelCommand cmd)
{
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_Not_U8_U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U1x8 p1)\n"
			"{\n"
			"  U8x8 r;\n"
			"  Convert_U8_U1(&r, p1);\n"
			"  *p0 = ~r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_Not_U1_U8(AgoNode * node, AgoKernelCommand cmd)
{
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_Not_U1_U8(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U8x8 p1)\n"
			"{\n"
			"  U1x8 r;\n"
			"  Convert_U1_U8(&r, p1);\n"
			"  *p0 = ~r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_Not_U1_U1(AgoNode * node, AgoKernelCommand cmd)
{
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_Not_U1_U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U1x8 p1)\n"
			"{\n"
			"  *p0 = ~p1;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_Lut_U8_U8(AgoNode * node, AgoKernelCommand cmd)
{
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iLut = node->paramList[2];
		if (HafCpu_Lut_U8_U8(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes, iLut->buffer)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, __read_only image1d_t lut)\n"
			"{\n"
			"    U8x8 r;\n"
			"    float4 f;\n"
			"    f.s0 = read_imagef(lut, (int)( p1.s0        & 255)).s0 * 255.0f;\n"
			"    f.s1 = read_imagef(lut, (int)((p1.s0 >>  8) & 255)).s0 * 255.0f;\n"
			"    f.s2 = read_imagef(lut, (int)((p1.s0 >> 16) & 255)).s0 * 255.0f;\n"
			"    f.s3 = read_imagef(lut, (int)( p1.s0 >> 24       )).s0 * 255.0f;\n"
			"    r.s0 = amd_pack(f);\n"
			"    f.s0 = read_imagef(lut, (int)( p1.s1        & 255)).s0 * 255.0f;\n"
			"    f.s1 = read_imagef(lut, (int)((p1.s1 >>  8) & 255)).s0 * 255.0f;\n"
			"    f.s2 = read_imagef(lut, (int)((p1.s1 >> 16) & 255)).s0 * 255.0f;\n"
			"    f.s3 = read_imagef(lut, (int)( p1.s1 >> 24       )).s0 * 255.0f;\n"
			"    r.s1 = amd_pack(f);\n"
			"    *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_Threshold_U8_U8_Binary(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iThr = node->paramList[2];
		if (HafCpu_Threshold_U8_U8_Binary(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes, iThr->u.thr.threshold_lower)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		if (!(status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8))) {
			if (node->paramList[2]->u.thr.thresh_type != VX_THRESHOLD_TYPE_BINARY || node->paramList[2]->u.thr.data_type != VX_TYPE_UINT8)
				return VX_ERROR_INVALID_TYPE;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_Threshold_U8_U8_Binary(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"#define %s Threshold_U8_U8_Binary\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_Threshold_U8_U8_Range(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iThr = node->paramList[2];
		if (HafCpu_Threshold_U8_U8_Range(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes, iThr->u.thr.threshold_lower, iThr->u.thr.threshold_upper)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		if (!(status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8))) {
			if (node->paramList[2]->u.thr.thresh_type != VX_THRESHOLD_TYPE_RANGE || node->paramList[2]->u.thr.data_type != VX_TYPE_UINT8)
				return VX_ERROR_INVALID_TYPE;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_Threshold_U8_U8_Range(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"#define %s Threshold_U8_U8_Range\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_Threshold_U1_U8_Binary(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iThr = node->paramList[2];
		if (HafCpu_Threshold_U1_U8_Binary(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes, iThr->u.thr.threshold_lower)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		if (!(status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8))) {
			if (node->paramList[2]->u.thr.thresh_type != VX_THRESHOLD_TYPE_BINARY || node->paramList[2]->u.thr.data_type != VX_TYPE_UINT8)
				return VX_ERROR_INVALID_TYPE;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_Threshold_U8_U8_Binary(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U8x8 p1, uint p2)\n"
			"{\n"
			"  U8x8 r1;\n"
			"  Threshold_U8_U8_Binary(&r1, p1, p2);\n"
			"  Convert_U1_U8(p0, r1);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_Threshold_U1_U8_Range(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iThr = node->paramList[2];
		if (HafCpu_Threshold_U1_U8_Range(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes, iThr->u.thr.threshold_lower, iThr->u.thr.threshold_upper)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		if (!(status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8))) {
			if (node->paramList[2]->u.thr.thresh_type != VX_THRESHOLD_TYPE_RANGE || node->paramList[2]->u.thr.data_type != VX_TYPE_UINT8)
				return VX_ERROR_INVALID_TYPE;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_Threshold_U8_U8_Range(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U8x8 p1, uint2 p2)\n"
			"{\n"
			"  U8x8 r1;\n"
			"  Threshold_U8_U8_Range(&r1, p1, p2);\n"
			"  Convert_U1_U8(p0, r1);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ThresholdNot_U8_U8_Binary(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iThr = node->paramList[2];
		if (HafCpu_ThresholdNot_U8_U8_Binary(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes, iThr->u.thr.threshold_lower)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		if (!(status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8))) {
			if (node->paramList[2]->u.thr.thresh_type != VX_THRESHOLD_TYPE_BINARY || node->paramList[2]->u.thr.data_type != VX_TYPE_UINT8)
				return VX_ERROR_INVALID_TYPE;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_Threshold_U8_U8_Binary(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, uint p2)\n"
			"{\n"
			"  U8x8 r1;\n"
			"  Threshold_U8_U8_Binary(&r1, p1, p2);\n"
			"  *p0 = ~r1;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ThresholdNot_U8_U8_Range(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iThr = node->paramList[2];
		if (HafCpu_ThresholdNot_U8_U8_Range(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes, iThr->u.thr.threshold_lower, iThr->u.thr.threshold_upper)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		if (!(status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8))) {
			if (node->paramList[2]->u.thr.thresh_type != VX_THRESHOLD_TYPE_RANGE || node->paramList[2]->u.thr.data_type != VX_TYPE_UINT8)
				return VX_ERROR_INVALID_TYPE;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_Threshold_U8_U8_Range(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, uint2 p2)\n"
			"{\n"
			"  U8x8 r1;\n"
			"  Threshold_U8_U8_Range(&r1, p1, p2);\n"
			"  *p0 = ~r1;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ThresholdNot_U1_U8_Binary(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iThr = node->paramList[2];
		if (HafCpu_ThresholdNot_U1_U8_Binary(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes, iThr->u.thr.threshold_lower)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		if (!(status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8))) {
			if (node->paramList[2]->u.thr.thresh_type != VX_THRESHOLD_TYPE_BINARY || node->paramList[2]->u.thr.data_type != VX_TYPE_UINT8)
				return VX_ERROR_INVALID_TYPE;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_Threshold_U8_U8_Binary(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U8x8 p1, uint p2)\n"
			"{\n"
			"  U8x8 r1;\n"
			"  Threshold_U8_U8_Binary(&r1, p1, p2);\n"
			"  Convert_U1_U8(p0, ~r1);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ThresholdNot_U1_U8_Range(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iThr = node->paramList[2];
		if (HafCpu_ThresholdNot_U1_U8_Range(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes, iThr->u.thr.threshold_lower, iThr->u.thr.threshold_upper)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		if (!(status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8))) {
			if (node->paramList[2]->u.thr.thresh_type != VX_THRESHOLD_TYPE_RANGE || node->paramList[2]->u.thr.data_type != VX_TYPE_UINT8)
				return VX_ERROR_INVALID_TYPE;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_Threshold_U8_U8_Range(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U8x8 p1, uint2 p2)\n"
			"{\n"
			"  U8x8 r1;\n"
			"  Threshold_U8_U8_Range(&r1, p1, p2);\n"
			"  Convert_U1_U8(p0, ~r1);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                    
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorDepth_U8_S16_Wrap(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		vx_int32 shift = node->paramList[2]->u.scalar.u.i;
		if (HafCpu_ColorDepth_U8_S16_Wrap(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, (vx_int16 *)iImg->buffer, iImg->u.img.stride_in_bytes, shift)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN_S(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_S16, VX_TYPE_INT32);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, S16x8 p1, uint p2)\n"
			"{\n"
			"  U8x8 r;\n"
			"  p2 += 16;\n"
			"  r.s0  = ((((int)p1.s0)  << 16) >> p2) & 0xff;\n"
			"  r.s0 |= ((((int)p1.s0)         >> p2) & 0xff) <<  8;\n"
			"  r.s0 |= (((((int)p1.s1) << 16) >> p2) & 0xff) << 16;\n"
			"  r.s0 |= ((((int)p1.s1)         >> p2) & 0xff) << 24;\n"
			"  r.s1  = ((((int)p1.s2)  << 16) >> p2) & 0xff;\n"
			"  r.s1 |= ((((int)p1.s2)         >> p2) & 0xff) <<  8;\n"
			"  r.s1 |= (((((int)p1.s3) << 16) >> p2) & 0xff) << 16;\n"
			"  r.s1 |= ((((int)p1.s3)         >> p2) & 0xff) << 24;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorDepth_U8_S16_Sat(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		vx_int32 shift = node->paramList[2]->u.scalar.u.i;
		if (HafCpu_ColorDepth_U8_S16_Sat(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, (vx_int16 *)iImg->buffer, iImg->u.img.stride_in_bytes, shift)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN_S(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_S16, VX_TYPE_INT32);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, S16x8 p1, uint p2)\n"
			"{\n"
			"  U8x8 r;\n"
			"  float4 f;\n"
			"  p2 += 16;\n"
			"  f.s0 = (float)((((int)p1.s0) << 16) >> p2);\n"
			"  f.s1 = (float)( ((int)p1.s0)        >> p2);\n"
			"  f.s2 = (float)((((int)p1.s1) << 16) >> p2);\n"
			"  f.s3 = (float)( ((int)p1.s1)        >> p2);\n"
			"  r.s0 = amd_pack(f);\n"
			"  f.s0 = (float)((((int)p1.s2) << 16) >> p2);\n"
			"  f.s1 = (float)( ((int)p1.s2)        >> p2);\n"
			"  f.s2 = (float)((((int)p1.s3) << 16) >> p2);\n"
			"  f.s3 = (float)( ((int)p1.s3)        >> p2);\n"
			"  r.s1 = amd_pack(f);\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorDepth_S16_U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		vx_int32 shift = node->paramList[2]->u.scalar.u.i;
		if (HafCpu_ColorDepth_S16_U8(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes, shift)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN_S(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8, VX_TYPE_INT32);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, U8x8 p1, uint p2)\n"
			"{\n"
			"  S16x8 r;\n"
			"  r.s0  =  (p1.s0 & 0x000000ff) <<     p2 ;\n"
			"  r.s0 |=  (p1.s0 & 0x0000ff00) << ( 8+p2);\n"
			"  r.s1  =  (p1.s0 & 0x00ff0000) >> (16-p2);\n"
			"  r.s1 |=  (p1.s0 & 0xff000000) >> ( 8-p2);\n"
			"  r.s2  =  (p1.s1 & 0x000000ff) <<     p2 ;\n"
			"  r.s2 |=  (p1.s1 & 0x0000ff00) << ( 8+p2);\n"
			"  r.s3  =  (p1.s1 & 0x00ff0000) >> (16-p2);\n"
			"  r.s3 |=  (p1.s1 & 0xff000000) >> ( 8-p2);\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_Add_U8_U8U8_Wrap(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Add_U8_U8U8_Wrap(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, U8x8 p2)\n"
			"{\n"
			"  U8x8 r;\n"
			"  r.s0  = (p1.s0 +  p2.s0              ) & 0x000000ff;\n"
			"  r.s0 |= (p1.s0 + (p2.s0 & 0x0000ff00)) & 0x0000ff00;\n"
			"  r.s0 |= (p1.s0 + (p2.s0 & 0x00ff0000)) & 0x00ff0000;\n"
			"  r.s0 |= (p1.s0 + (p2.s0 & 0xff000000)) & 0xff000000;\n"
			"  r.s1  = (p1.s1 +  p2.s1              ) & 0x000000ff;\n"
			"  r.s1 |= (p1.s1 + (p2.s1 & 0x0000ff00)) & 0x0000ff00;\n"
			"  r.s1 |= (p1.s1 + (p2.s1 & 0x00ff0000)) & 0x00ff0000;\n"
			"  r.s1 |= (p1.s1 + (p2.s1 & 0xff000000)) & 0xff000000;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Add_U8_U8U8_Sat(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Add_U8_U8U8_Sat(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, U8x8 p2)\n"
			"{\n"
			"  U8x8 r;\n"
			"  r.s0 = amd_pack(amd_unpack(p1.s0) + amd_unpack(p2.s0));\n"
			"  r.s1 = amd_pack(amd_unpack(p1.s1) + amd_unpack(p2.s1));\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Sub_U8_U8U8_Wrap(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Sub_U8_U8U8_Wrap(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, U8x8 p2)\n"
			"{\n"
			"  U8x8 r;\n"
			"  r.s0  = (p1.s0 -  p2.s0              ) & 0x000000ff;\n"
			"  r.s0 |= (p1.s0 - (p2.s0 & 0x0000ff00)) & 0x0000ff00;\n"
			"  r.s0 |= (p1.s0 - (p2.s0 & 0x00ff0000)) & 0x00ff0000;\n"
			"  r.s0 |= (p1.s0 - (p2.s0 & 0xff000000)) & 0xff000000;\n"
			"  r.s1  = (p1.s1 -  p2.s1              ) & 0x000000ff;\n"
			"  r.s1 |= (p1.s1 - (p2.s1 & 0x0000ff00)) & 0x0000ff00;\n"
			"  r.s1 |= (p1.s1 - (p2.s1 & 0x00ff0000)) & 0x00ff0000;\n"
			"  r.s1 |= (p1.s1 - (p2.s1 & 0xff000000)) & 0xff000000;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Sub_U8_U8U8_Sat(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Sub_U8_U8U8_Sat(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, U8x8 p2)\n"
			"{\n"
			"  U8x8 r;\n"
			"  r.s0 = amd_pack(amd_unpack(p1.s0) - amd_unpack(p2.s0));\n"
			"  r.s1 = amd_pack(amd_unpack(p1.s1) - amd_unpack(p2.s1));\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Mul_U8_U8U8_Wrap_Trunc(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		vx_float32 scale = node->paramList[3]->u.scalar.u.f;
		if (HafCpu_Mul_U8_U8U8_Wrap_Trunc(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes, scale)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN_S(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_TYPE_FLOAT32);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, U8x8 p2, float p3)\n"
			"{\n"
			"  U8x8 r;\n"
			"  r.s0  = ((int)(p3 * amd_unpack0(p1.s0) * amd_unpack0(p2.s0)) & 0x000000ff)      ;\n"
			"  r.s0 |= ((int)(p3 * amd_unpack1(p1.s0) * amd_unpack1(p2.s0)) & 0x000000ff) <<  8;\n"
			"  r.s0 |= ((int)(p3 * amd_unpack2(p1.s0) * amd_unpack2(p2.s0)) & 0x000000ff) << 16;\n"
			"  r.s0 |= ((int)(p3 * amd_unpack3(p1.s0) * amd_unpack3(p2.s0))             ) << 24;\n"
			"  r.s1  = ((int)(p3 * amd_unpack0(p1.s1) * amd_unpack0(p2.s1)) & 0x000000ff)      ;\n"
			"  r.s1 |= ((int)(p3 * amd_unpack1(p1.s1) * amd_unpack1(p2.s1)) & 0x000000ff) <<  8;\n"
			"  r.s1 |= ((int)(p3 * amd_unpack2(p1.s1) * amd_unpack2(p2.s1)) & 0x000000ff) << 16;\n"
			"  r.s1 |= ((int)(p3 * amd_unpack3(p1.s1) * amd_unpack3(p2.s1))             ) << 24;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Mul_U8_U8U8_Wrap_Round(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		vx_float32 scale = node->paramList[3]->u.scalar.u.f;
		if (HafCpu_Mul_U8_U8U8_Wrap_Round(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes, scale)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN_S(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_TYPE_FLOAT32);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, U8x8 p2, float p3)\n"
			"{\n"
			"  U8x8 r;\n"
			"  r.s0  = ((int)(p3 * amd_unpack0(p1.s0) * amd_unpack0(p2.s0) + (0.5f - 0.00006103515625f)) & 0x000000ff)      ;\n"
			"  r.s0 |= ((int)(p3 * amd_unpack1(p1.s0) * amd_unpack1(p2.s0) + (0.5f - 0.00006103515625f)) & 0x000000ff) <<  8;\n"
			"  r.s0 |= ((int)(p3 * amd_unpack2(p1.s0) * amd_unpack2(p2.s0) + (0.5f - 0.00006103515625f)) & 0x000000ff) << 16;\n"
			"  r.s0 |= ((int)(p3 * amd_unpack3(p1.s0) * amd_unpack3(p2.s0) + (0.5f - 0.00006103515625f))             ) << 24;\n"
			"  r.s1  = ((int)(p3 * amd_unpack0(p1.s1) * amd_unpack0(p2.s1) + (0.5f - 0.00006103515625f)) & 0x000000ff)      ;\n"
			"  r.s1 |= ((int)(p3 * amd_unpack1(p1.s1) * amd_unpack1(p2.s1) + (0.5f - 0.00006103515625f)) & 0x000000ff) <<  8;\n"
			"  r.s1 |= ((int)(p3 * amd_unpack2(p1.s1) * amd_unpack2(p2.s1) + (0.5f - 0.00006103515625f)) & 0x000000ff) << 16;\n"
			"  r.s1 |= ((int)(p3 * amd_unpack3(p1.s1) * amd_unpack3(p2.s1) + (0.5f - 0.00006103515625f))             ) << 24;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Mul_U8_U8U8_Sat_Trunc(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		vx_float32 scale = node->paramList[3]->u.scalar.u.f;
		if (HafCpu_Mul_U8_U8U8_Sat_Trunc(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes, scale)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN_S(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_TYPE_FLOAT32);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, U8x8 p2, float p3)\n"
			"{\n"
			"  U8x8 r;\n"
			"  float4 f;\n"
			"  f.s0 = p3 * amd_unpack0(p1.s0) * amd_unpack0(p2.s0) - (0.5f - 0.00006103515625f);\n"
			"  f.s1 = p3 * amd_unpack1(p1.s0) * amd_unpack1(p2.s0) - (0.5f - 0.00006103515625f);\n"
			"  f.s2 = p3 * amd_unpack2(p1.s0) * amd_unpack2(p2.s0) - (0.5f - 0.00006103515625f);\n"
			"  f.s3 = p3 * amd_unpack3(p1.s0) * amd_unpack3(p2.s0) - (0.5f - 0.00006103515625f);\n"
			"  r.s0 = amd_pack(f);\n"
			"  f.s0 = p3 * amd_unpack0(p1.s1) * amd_unpack0(p2.s1) - (0.5f - 0.00006103515625f);\n"
			"  f.s1 = p3 * amd_unpack1(p1.s1) * amd_unpack1(p2.s1) - (0.5f - 0.00006103515625f);\n"
			"  f.s2 = p3 * amd_unpack2(p1.s1) * amd_unpack2(p2.s1) - (0.5f - 0.00006103515625f);\n"
			"  f.s3 = p3 * amd_unpack3(p1.s1) * amd_unpack3(p2.s1) - (0.5f - 0.00006103515625f);\n"
			"  r.s1 = amd_pack(f);\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Mul_U8_U8U8_Sat_Round(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		vx_float32 scale = node->paramList[3]->u.scalar.u.f;
		if (HafCpu_Mul_U8_U8U8_Sat_Round(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes, scale)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN_S(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_TYPE_FLOAT32);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, U8x8 p2, float p3)\n"
			"{\n"
			"  U8x8 r;\n"
			"  float4 f;\n"
			"  f.s0 = p3 * amd_unpack0(p1.s0) * amd_unpack0(p2.s0);\n"
			"  f.s1 = p3 * amd_unpack1(p1.s0) * amd_unpack1(p2.s0);\n"
			"  f.s2 = p3 * amd_unpack2(p1.s0) * amd_unpack2(p2.s0);\n"
			"  f.s3 = p3 * amd_unpack3(p1.s0) * amd_unpack3(p2.s0);\n"
			"  r.s0 = amd_pack(f);\n"
			"  f.s0 = p3 * amd_unpack0(p1.s1) * amd_unpack0(p2.s1);\n"
			"  f.s1 = p3 * amd_unpack1(p1.s1) * amd_unpack1(p2.s1);\n"
			"  f.s2 = p3 * amd_unpack2(p1.s1) * amd_unpack2(p2.s1);\n"
			"  f.s3 = p3 * amd_unpack3(p1.s1) * amd_unpack3(p2.s1);\n"
			"  r.s1 = amd_pack(f);\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_And_U8_U8U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_And_U8_U8U8(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, U8x8 p2)\n"
			"{\n"
			"  *p0 = p1 & p2;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_And_U8_U8U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_And_U8_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, U1x8 p2)\n"
			"{\n"
			"  U8x8 r;\n"
			"  Convert_U8_U1(&r, p2);\n"
			"  *p0 = p1 & r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_And_U8_U1U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[2];
		AgoData * iImg1 = node->paramList[1];
		if (HafCpu_And_U8_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U1x8 p1, U8x8 p2)\n"
			"{\n"
			"  U8x8 r;\n"
			"  Convert_U8_U1(&r, p1);\n"
			"  *p0 = p2 & r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_And_U8_U1U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_And_U8_U1U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U1x8 p1, U1x8 p2)\n"
			"{\n"
			"  U8x8 r1, r2;\n"
			"  Convert_U8_U1(&r1, p1);\n"
			"  Convert_U8_U1(&r2, p2);\n"
			"  *p0 = r1 & r2;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_And_U1_U8U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_And_U1_U8U8(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U8x8 p1, U8x8 p2)\n"
			"{\n"
			"  U8x8 r = p1 & p2;\n"
			"  Convert_U1_U8(p0, r);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_And_U1_U8U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_And_U1_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U8x8 p1, U1x8 p2)\n"
			"{\n"
			"  U1x8 r;\n"
			"  Convert_U1_U8(&r, p1);\n"
			"  *p0 = r & p2;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_And_U1_U1U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[2];
		AgoData * iImg1 = node->paramList[1];
		if (HafCpu_And_U1_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U1x8 p1, U8x8 p2)\n"
			"{\n"
			"  U1x8 r;\n"
			"  Convert_U1_U8(&r, p2);\n"
			"  *p0 = r & p1;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_And_U1_U1U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_And_U1_U1U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U1x8 p1, U1x8 p2)\n"
			"{\n"
			"  *p0 = p1 & p2;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Or_U8_U8U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Or_U8_U8U8(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, U8x8 p2)\n"
			"{\n"
			"  *p0 = p1 | p2;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Or_U8_U8U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Or_U8_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, U1x8 p2)\n"
			"{\n"
			"  U8x8 r;\n"
			"  Convert_U8_U1(&r, p2);\n"
			"  *p0 = p1 | r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Or_U8_U1U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[2];
		AgoData * iImg1 = node->paramList[1];
		if (HafCpu_Or_U8_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U1x8 p1, U8x8 p2)\n"
			"{\n"
			"  U8x8 r;\n"
			"  Convert_U8_U1(&r, p1);\n"
			"  *p0 = p2 | r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Or_U8_U1U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Or_U8_U1U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U1x8 p1, U1x8 p2)\n"
			"{\n"
			"  U8x8 r1, r2;\n"
			"  Convert_U8_U1(&r1, p1);\n"
			"  Convert_U8_U1(&r2, p2);\n"
			"  *p0 = r1 | r2;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Or_U1_U8U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Or_U1_U8U8(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U8x8 p1, U8x8 p2)\n"
			"{\n"
			"  U8x8 r = p1 | p2;\n"
			"  Convert_U1_U8(p0, r);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Or_U1_U8U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Or_U1_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U8x8 p1, U1x8 p2)\n"
			"{\n"
			"  U1x8 r;\n"
			"  Convert_U1_U8(&r, p1);\n"
			"  *p0 = r | p2;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Or_U1_U1U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[2];
		AgoData * iImg1 = node->paramList[1];
		if (HafCpu_Or_U1_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U1x8 p1, U8x8 p2)\n"
			"{\n"
			"  U1x8 r;\n"
			"  Convert_U1_U8(&r, p2);\n"
			"  *p0 = r | p1;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Or_U1_U1U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Or_U1_U1U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U1x8 p1, U1x8 p2)\n"
			"{\n"
			"  *p0 = p1 | p2;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Xor_U8_U8U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Xor_U8_U8U8(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, U8x8 p2)\n"
			"{\n"
			"  *p0 = p1 ^ p2;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Xor_U8_U8U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Xor_U8_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, U1x8 p2)\n"
			"{\n"
			"  U8x8 r;\n"
			"  Convert_U8_U1(&r, p2);\n"
			"  *p0 = p1 ^ r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Xor_U8_U1U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[2];
		AgoData * iImg1 = node->paramList[1];
		if (HafCpu_Xor_U8_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U1x8 p1, U8x8 p2)\n"
			"{\n"
			"  U8x8 r;\n"
			"  Convert_U8_U1(&r, p1);\n"
			"  *p0 = p2 ^ r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Xor_U8_U1U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Xor_U8_U1U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U1x8 p1, U1x8 p2)\n"
			"{\n"
			"  U8x8 r1, r2;\n"
			"  Convert_U8_U1(&r1, p1);\n"
			"  Convert_U8_U1(&r2, p2);\n"
			"  *p0 = r1 ^ r2;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Xor_U1_U8U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Xor_U1_U8U8(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U8x8 p1, U8x8 p2)\n"
			"{\n"
			"  U8x8 r = p1 ^ p2;\n"
			"  Convert_U1_U8(p0, r);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Xor_U1_U8U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Xor_U1_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U8x8 p1, U1x8 p2)\n"
			"{\n"
			"  U1x8 r;\n"
			"  Convert_U1_U8(&r, p1);\n"
			"  *p0 = r ^ p2;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Xor_U1_U1U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[2];
		AgoData * iImg1 = node->paramList[1];
		if (HafCpu_Xor_U1_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U1x8 p1, U8x8 p2)\n"
			"{\n"
			"  U1x8 r;\n"
			"  Convert_U1_U8(&r, p2);\n"
			"  *p0 = r ^ p1;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Xor_U1_U1U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Xor_U1_U1U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U1x8 p1, U1x8 p2)\n"
			"{\n"
			"  *p0 = p1 ^ p2;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Nand_U8_U8U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Nand_U8_U8U8(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, U8x8 p2)\n"
			"{\n"
			"  U8x8 r;\n"
			"  r = p1 & p2;\n"
			"  *p0 = ~r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Nand_U8_U8U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Nand_U8_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, U1x8 p2)\n"
			"{\n"
			"  U8x8 r;\n"
			"  Convert_U8_U1(&r, p2);\n"
			"  r = r & p1;\n"
			"  *p0 = ~r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Nand_U8_U1U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[2];
		AgoData * iImg1 = node->paramList[1];
		if (HafCpu_Nand_U8_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U1x8 p1, U8x8 p2)\n"
			"{\n"
			"  U8x8 r;\n"
			"  Convert_U8_U1(&r, p1);\n"
			"  r = r & p2;\n"
			"  *p0 = ~r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Nand_U8_U1U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Nand_U8_U1U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U1x8 p1, U1x8 p2)\n"
			"{\n"
			"  U8x8 r1, r2;\n"
			"  Convert_U8_U1(&r1, p1);\n"
			"  Convert_U8_U1(&r2, p2);\n"
			"  r1 = r1 & r2;\n"
			"  *p0 = ~r1;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Nand_U1_U8U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Nand_U1_U8U8(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U8x8 p1, U8x8 p2)\n"
			"{\n"
			"  U1x8 r;\n"
			"  p1 = p1 & p2;\n"
			"  Convert_U1_U8(&r, p1);\n"
			"  *p0 = ~r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Nand_U1_U8U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Nand_U1_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U8x8 p1, U1x8 p2)\n"
			"{\n"
			"  U1x8 r;\n"
			"  Convert_U1_U8(&r, p1);\n"
			"  *p0 = ~(r & p2);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Nand_U1_U1U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[2];
		AgoData * iImg1 = node->paramList[1];
		if (HafCpu_Nand_U1_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U1x8 p1, U8x8 p2)\n"
			"{\n"
			"  U1x8 r;\n"
			"  Convert_U1_U8(&r, p2);\n"
			"  *p0 = ~(r & p1);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Nand_U1_U1U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Nand_U1_U1U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U1x8 p1, U1x8 p2)\n"
			"{\n"
			"  *p0 = ~(p1 & p2);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Nor_U8_U8U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Nor_U8_U8U8(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, U8x8 p2)\n"
			"{\n"
			"  U8x8 r;\n"
			"  *p0 = ~(p1 | p2);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Nor_U8_U8U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Nor_U8_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, U1x8 p2)\n"
			"{\n"
			"  U8x8 r;\n"
			"  Convert_U8_U1(&r, p2);\n"
			"  *p0 = ~(r | p1);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Nor_U8_U1U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[2];
		AgoData * iImg1 = node->paramList[1];
		if (HafCpu_Nor_U8_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U1x8 p1, U8x8 p2)\n"
			"{\n"
			"  U8x8 r;\n"
			"  Convert_U8_U1(&r, p1);\n"
			"  *p0 = ~(r | p2);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Nor_U8_U1U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Nor_U8_U1U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U1x8 p1, U1x8 p2)\n"
			"{\n"
			"  U8x8 r1, r2;\n"
			"  Convert_U8_U1(&r1, p1);\n"
			"  Convert_U8_U1(&r2, p2);\n"
			"  *p0 = ~(r1 | r2);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Nor_U1_U8U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Nor_U1_U8U8(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U8x8 p1, U8x8 p2)\n"
			"{\n"
			"  U1x8 r;\n"
			"  Convert_U1_U8(&r, p1 | p2);\n"
			"  *p0 = ~r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Nor_U1_U8U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Nor_U1_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U8x8 p1, U1x8 p2)\n"
			"{\n"
			"  U1x8 r;\n"
			"  Convert_U1_U8(&r, p1);\n"
			"  *p0 = ~(r | p2);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Nor_U1_U1U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[2];
		AgoData * iImg1 = node->paramList[1];
		if (HafCpu_Nor_U1_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U1x8 p1, U8x8 p2)\n"
			"{\n"
			"  U1x8 r;\n"
			"  Convert_U1_U8(&r, p2);\n"
			"  *p0 = ~(r | p1);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Nor_U1_U1U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Nor_U1_U1U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U1x8 p1, U1x8 p2)\n"
			"{\n"
			"  *p0 = ~(p1 | p2);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Xnor_U8_U8U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Xnor_U8_U8U8(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, U8x8 p2)\n"
			"{\n"
			"  *p0 = ~(p1 ^ p2);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Xnor_U8_U8U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Xnor_U8_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, U1x8 p2)\n"
			"{\n"
			"  U8x8 r;\n"
			"  Convert_U8_U1(&r, p2);\n"
			"  *p0 = ~(r ^ p1);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Xnor_U8_U1U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[2];
		AgoData * iImg1 = node->paramList[1];
		if (HafCpu_Xnor_U8_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U1x8 p1, U8x8 p2)\n"
			"{\n"
			"  U8x8 r;\n"
			"  Convert_U8_U1(&r, p1);\n"
			"  *p0 = ~(r ^ p2);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Xnor_U8_U1U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Xnor_U8_U1U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U1x8 p1, U1x8 p2)\n"
			"{\n"
			"  U8x8 r1, r2;\n"
			"  Convert_U8_U1(&r1, p1);\n"
			"  Convert_U8_U1(&r2, p2);\n"
			"  *p0 = ~(r1 ^ r2);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Xnor_U1_U8U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Xnor_U1_U8U8(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U8x8 p1, U8x8 p2)\n"
			"{\n"
			"  Convert_U1_U8(p0, ~(p1 ^ p2));\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Xnor_U1_U8U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Xnor_U1_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U8x8 p1, U1x8 p2)\n"
			"{\n"
			"  U1x8 r;\n"
			"  Convert_U1_U8(&r, p1);\n"
			"  *p0 = ~(r ^ p2);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Xnor_U1_U1U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[2];
		AgoData * iImg1 = node->paramList[1];
		if (HafCpu_Xnor_U1_U8U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U1x8 p1, U8x8 p2)\n"
			"{\n"
			"  U1x8 r;\n"
			"  Convert_U1_U8(&r, p2);\n"
			"  *p0 = ~(r ^ p1);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Xnor_U1_U1U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Xnor_U1_U1U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U1x8 p1, U1x8 p2)\n"
			"{\n"
			"  *p0 = ~(p1 ^ p2);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_AbsDiff_U8_U8U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_AbsDiff_U8_U8U8(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, U8x8 p2)\n"
			"{\n"
			"  U8x8 r;\n"
			"  r.s0 = amd_pack(fabs(amd_unpack(p1.s0) - amd_unpack(p2.s0)));\n"
			"  r.s1 = amd_pack(fabs(amd_unpack(p1.s1) - amd_unpack(p2.s1)));\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_AccumulateWeighted_U8_U8U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		vx_float32 alpha = node->paramList[2]->u.scalar.u.f;
		if (HafCpu_AccumulateWeighted_U8_U8U8(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes, alpha)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8 || node->paramList[1]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != node->paramList[1]->u.img.width || height != node->paramList[1]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (node->paramList[2]->u.scalar.type != VX_TYPE_FLOAT32)
			return VX_ERROR_INVALID_TYPE;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1, float p2)\n"
			"{\n"
			"  U8x8 r = *p0;\n"
			"  float p3 = 1.0f - p2;\n"
			"  float4 f;\n"
			"  f.s0 = p3 * amd_unpack0(r.s0) + p2 * amd_unpack0(p1.s0);\n"
			"  f.s1 = p3 * amd_unpack1(r.s0) + p2 * amd_unpack1(p1.s0);\n"
			"  f.s2 = p3 * amd_unpack2(r.s0) + p2 * amd_unpack2(p1.s0);\n"
			"  f.s3 = p3 * amd_unpack3(r.s0) + p2 * amd_unpack3(p1.s0);\n"
			"  r.s0 = amd_pack(f);\n"
			"  f.s0 = p3 * amd_unpack0(r.s1) + p2 * amd_unpack0(p1.s1);\n"
			"  f.s1 = p3 * amd_unpack1(r.s1) + p2 * amd_unpack1(p1.s1);\n"
			"  f.s2 = p3 * amd_unpack2(r.s1) + p2 * amd_unpack2(p1.s1);\n"
			"  f.s3 = p3 * amd_unpack3(r.s1) + p2 * amd_unpack3(p1.s1);\n"
			"  r.s1 = amd_pack(f);\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[0];
		AgoData * inp2 = node->paramList[1];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Add_S16_U8U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Add_S16_U8U8(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, U8x8 p1, U8x8 p2)\n"
			"{\n"
			"  S16x8 r;\n"
			"  r.s0  = ((p1.s0 & 0x000000ff) + (p2.s0 & 0x000000ff));\n"
			"  r.s0 |= ((p1.s0 & 0x0000ff00) + (p2.s0 & 0x0000ff00)) <<  8;\n"
			"  r.s1  = ((p1.s0 & 0x00ff0000) + (p2.s0 & 0x00ff0000)) >> 16;\n"
			"  r.s1 |= ((p1.s0 >>        24) + (p2.s0 >>        24)) << 16;\n"
			"  r.s2  = ((p1.s1 & 0x000000ff) + (p2.s1 & 0x000000ff));\n"
			"  r.s2 |= ((p1.s1 & 0x0000ff00) + (p2.s1 & 0x0000ff00)) <<  8;\n"
			"  r.s3  = ((p1.s1 & 0x00ff0000) + (p2.s1 & 0x00ff0000)) >> 16;\n"
			"  r.s3 |= ((p1.s1 >>        24) + (p2.s1 >>        24)) << 16;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Sub_S16_U8U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Sub_S16_U8U8(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, U8x8 p1, U8x8 p2)\n"
			"{\n"
			"  S16x8 r;\n"
			"  r.s0  = ((p1.s0 & 0x000000ff) - (p2.s0 & 0x000000ff)) & 0x0000ffff;\n"
			"  r.s0 |= ((p1.s0 & 0x0000ff00) - (p2.s0 & 0x0000ff00)) <<  8;\n"
			"  r.s1  = ((p1.s0 & 0x00ff0000) - (p2.s0 & 0x00ff0000)) >> 16;\n"
			"  r.s1 |= ((p1.s0 >>        24) - (p2.s0 >>        24)) << 16;\n"
			"  r.s2  = ((p1.s1 & 0x000000ff) - (p2.s1 & 0x000000ff)) & 0x0000ffff;\n"
			"  r.s2 |= ((p1.s1 & 0x0000ff00) - (p2.s1 & 0x0000ff00)) <<  8;\n"
			"  r.s3  = ((p1.s1 & 0x00ff0000) - (p2.s1 & 0x00ff0000)) >> 16;\n"
			"  r.s3 |= ((p1.s1 >>        24) - (p2.s1 >>        24)) << 16;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Mul_S16_U8U8_Wrap_Trunc(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		vx_float32 scale = node->paramList[3]->u.scalar.u.f;
		if (HafCpu_Mul_S16_U8U8_Wrap_Trunc(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes, scale)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN_S(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_TYPE_FLOAT32);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, U8x8 p1, U8x8 p2, float p3)\n"
			"{\n"
			"  S16x8 r;\n"
			"  r.s0  = (((int)(p3 * amd_unpack0(p1.s0) * amd_unpack0(p2.s0))) & 0x0000ffff)      ;\n"
			"  r.s0 |= (((int)(p3 * amd_unpack1(p1.s0) * amd_unpack1(p2.s0)))             ) << 16;\n"
			"  r.s1  = (((int)(p3 * amd_unpack2(p1.s0) * amd_unpack2(p2.s0))) & 0x0000ffff)      ;\n"
			"  r.s1 |= (((int)(p3 * amd_unpack3(p1.s0) * amd_unpack3(p2.s0)))             ) << 16;\n"
			"  r.s2  = (((int)(p3 * amd_unpack0(p1.s1) * amd_unpack0(p2.s1))) & 0x0000ffff)      ;\n"
			"  r.s2 |= (((int)(p3 * amd_unpack1(p1.s1) * amd_unpack1(p2.s1)))             ) << 16;\n"
			"  r.s3  = (((int)(p3 * amd_unpack2(p1.s1) * amd_unpack2(p2.s1))) & 0x0000ffff)      ;\n"
			"  r.s3 |= (((int)(p3 * amd_unpack3(p1.s1) * amd_unpack3(p2.s1)))             ) << 16;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Mul_S16_U8U8_Wrap_Round(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		vx_float32 scale = node->paramList[3]->u.scalar.u.f;
		if (HafCpu_Mul_S16_U8U8_Wrap_Round(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes, scale)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN_S(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_TYPE_FLOAT32);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, U8x8 p1, U8x8 p2, float p3)\n"
			"{\n"
			"  S16x8 r;\n"
			"  r.s0  = (((int)(p3 * amd_unpack0(p1.s0) * amd_unpack0(p2.s0) + 0.5f)) & 0x0000ffff)      ;\n"
			"  r.s0 |= (((int)(p3 * amd_unpack1(p1.s0) * amd_unpack1(p2.s0) + 0.5f))             ) << 16;\n"
			"  r.s1  = (((int)(p3 * amd_unpack2(p1.s0) * amd_unpack2(p2.s0) + 0.5f)) & 0x0000ffff)      ;\n"
			"  r.s1 |= (((int)(p3 * amd_unpack3(p1.s0) * amd_unpack3(p2.s0) + 0.5f))             ) << 16;\n"
			"  r.s2  = (((int)(p3 * amd_unpack0(p1.s1) * amd_unpack0(p2.s1) + 0.5f)) & 0x0000ffff)      ;\n"
			"  r.s2 |= (((int)(p3 * amd_unpack1(p1.s1) * amd_unpack1(p2.s1) + 0.5f))             ) << 16;\n"
			"  r.s3  = (((int)(p3 * amd_unpack2(p1.s1) * amd_unpack2(p2.s1) + 0.5f)) & 0x0000ffff)      ;\n"
			"  r.s3 |= (((int)(p3 * amd_unpack3(p1.s1) * amd_unpack3(p2.s1) + 0.5f))             ) << 16;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Mul_S16_U8U8_Sat_Trunc(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		vx_float32 scale = node->paramList[3]->u.scalar.u.f;
		if (HafCpu_Mul_S16_U8U8_Sat_Trunc(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes, scale)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN_S(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_TYPE_FLOAT32);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, U8x8 p1, U8x8 p2, float p3)\n"
			"{\n"
			"  S16x8 r;\n"
			"  r.s0  = (((int)(clamp(p3 * amd_unpack0(p1.s0) * amd_unpack0(p2.s0), -32768.0f, 32767.0f))) & 0x0000ffff)      ;\n"
			"  r.s0 |= (((int)(clamp(p3 * amd_unpack1(p1.s0) * amd_unpack1(p2.s0), -32768.0f, 32767.0f)))             ) << 16;\n"
			"  r.s1  = (((int)(clamp(p3 * amd_unpack2(p1.s0) * amd_unpack2(p2.s0), -32768.0f, 32767.0f))) & 0x0000ffff)      ;\n"
			"  r.s1 |= (((int)(clamp(p3 * amd_unpack3(p1.s0) * amd_unpack3(p2.s0), -32768.0f, 32767.0f)))             ) << 16;\n"
			"  r.s2  = (((int)(clamp(p3 * amd_unpack0(p1.s1) * amd_unpack0(p2.s1), -32768.0f, 32767.0f))) & 0x0000ffff)      ;\n"
			"  r.s2 |= (((int)(clamp(p3 * amd_unpack1(p1.s1) * amd_unpack1(p2.s1), -32768.0f, 32767.0f)))             ) << 16;\n"
			"  r.s3  = (((int)(clamp(p3 * amd_unpack2(p1.s1) * amd_unpack2(p2.s1), -32768.0f, 32767.0f))) & 0x0000ffff)      ;\n"
			"  r.s3 |= (((int)(clamp(p3 * amd_unpack3(p1.s1) * amd_unpack3(p2.s1), -32768.0f, 32767.0f)))             ) << 16;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Mul_S16_U8U8_Sat_Round(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		vx_float32 scale = node->paramList[3]->u.scalar.u.f;
		if (HafCpu_Mul_S16_U8U8_Sat_Round(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes, scale)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN_S(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_TYPE_FLOAT32);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, U8x8 p1, U8x8 p2, float p3)\n"
			"{\n"
			"    S16x8 r;\n"
			"    r.s0  = (((int)(clamp(p3 * amd_unpack0(p1.s0) * amd_unpack0(p2.s0) + 0.5f, -32768.0f, 32767.0f))) & 0x0000ffff)      ;\n"
			"    r.s0 |= (((int)(clamp(p3 * amd_unpack1(p1.s0) * amd_unpack1(p2.s0) + 0.5f, -32768.0f, 32767.0f)))             ) << 16;\n"
			"    r.s1  = (((int)(clamp(p3 * amd_unpack2(p1.s0) * amd_unpack2(p2.s0) + 0.5f, -32768.0f, 32767.0f))) & 0x0000ffff)      ;\n"
			"    r.s1 |= (((int)(clamp(p3 * amd_unpack3(p1.s0) * amd_unpack3(p2.s0) + 0.5f, -32768.0f, 32767.0f)))             ) << 16;\n"
			"    r.s2  = (((int)(clamp(p3 * amd_unpack0(p1.s1) * amd_unpack0(p2.s1) + 0.5f, -32768.0f, 32767.0f))) & 0x0000ffff)      ;\n"
			"    r.s2 |= (((int)(clamp(p3 * amd_unpack1(p1.s1) * amd_unpack1(p2.s1) + 0.5f, -32768.0f, 32767.0f)))             ) << 16;\n"
			"    r.s3  = (((int)(clamp(p3 * amd_unpack2(p1.s1) * amd_unpack2(p2.s1) + 0.5f, -32768.0f, 32767.0f))) & 0x0000ffff)      ;\n"
			"    r.s3 |= (((int)(clamp(p3 * amd_unpack3(p1.s1) * amd_unpack3(p2.s1) + 0.5f, -32768.0f, 32767.0f)))             ) << 16;\n"
			"    *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Add_S16_S16U8_Wrap(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Add_S16_S16U8_Wrap(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, (vx_int16 *)iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, S16x8 p1, U8x8 p2)\n"
			"{\n"
			"  S16x8 r;\n"
			"  r.s0  = ((((int)(p1.s0) << 16) >> 16) + ( p2.s0        & 0x000000ff)) & 0x0000ffff;\n"
			"  r.s0 |= ((       p1.s0  & 0xffff0000) + ((p2.s0 <<  8) & 0x00ff0000));\n"
			"  r.s1  = ((((int)(p1.s1) << 16) >> 16) + ((p2.s0 >> 16) & 0x000000ff)) & 0x0000ffff;\n"
			"  r.s1 |= ((       p1.s1  & 0xffff0000) + ((p2.s0 >>  8) & 0x00ff0000));\n"
			"  r.s2  = ((((int)(p1.s2) << 16) >> 16) + ( p2.s1        & 0x000000ff)) & 0x0000ffff;\n"
			"  r.s2 |= ((       p1.s2  & 0xffff0000) + ((p2.s1 <<  8) & 0x00ff0000));\n"
			"  r.s3  = ((((int)(p1.s3) << 16) >> 16) + ((p2.s1 >> 16) & 0x000000ff)) & 0x0000ffff;\n"
			"  r.s3 |= ((       p1.s3  & 0xffff0000) + ((p2.s1 >>  8) & 0x00ff0000));\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Add_S16_S16U8_Sat(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Add_S16_S16U8_Sat(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, (vx_int16 *)iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_Add_S16_S16U8_Sat(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"#define %s Add_S16_S16U8_Sat\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Accumulate_S16_S16U8_Sat(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_Accumulate_S16_S16U8_Sat(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_S16 || node->paramList[1]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != node->paramList[1]->u.img.width || height != node->paramList[1]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_Add_S16_S16U8_Sat(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, U8x8 p1)\n"
			"{\n"
			"  Add_S16_S16U8_Sat (p0, *p0, p1);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[0];
		AgoData * inp2 = node->paramList[1];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Sub_S16_S16U8_Wrap(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Sub_S16_S16U8_Wrap(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, (vx_int16 *)iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, S16x8 p1, U8x8 p2)\n"
			"{\n"
			"  S16x8 r;\n"
			"  r.s0  = ((((int)(p1.s0) << 16) >> 16) - ( p2.s0        & 0x000000ff)) & 0x0000ffff;\n"
			"  r.s0 |= ((       p1.s0  & 0xffff0000) - ((p2.s0 <<  8) & 0x00ff0000));\n"
			"  r.s1  = ((((int)(p1.s1) << 16) >> 16) - ((p2.s0 >> 16) & 0x000000ff)) & 0x0000ffff;\n"
			"  r.s1 |= ((       p1.s1  & 0xffff0000) - ((p2.s0 >>  8) & 0x00ff0000));\n"
			"  r.s2  = ((((int)(p1.s2) << 16) >> 16) - ( p2.s1        & 0x000000ff)) & 0x0000ffff;\n"
			"  r.s2 |= ((       p1.s2  & 0xffff0000) - ((p2.s1 <<  8) & 0x00ff0000));\n"
			"  r.s3  = ((((int)(p1.s3) << 16) >> 16) - ((p2.s1 >> 16) & 0x000000ff)) & 0x0000ffff;\n"
			"  r.s3 |= ((       p1.s3  & 0xffff0000) - ((p2.s1 >>  8) & 0x00ff0000));\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Sub_S16_S16U8_Sat(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Sub_S16_S16U8_Sat(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, (vx_int16 *)iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, S16x8 p1, U8x8 p2)\n"
			"{\n"
			"  S16x8 r;\n"
			"  r.s0  = (int)(clamp((float)(((int)(p1.s0) << 16) >> 16) - amd_unpack0(p2.s0), -32768.0f, 32767.0f)) & 0x0000ffff;\n"
			"  r.s0 |= (int)(clamp((float)( (int)(p1.s0)        >> 16) - amd_unpack1(p2.s0), -32768.0f, 32767.0f)) << 16;\n"
			"  r.s1  = (int)(clamp((float)(((int)(p1.s1) << 16) >> 16) - amd_unpack2(p2.s0), -32768.0f, 32767.0f)) & 0x0000ffff;\n"
			"  r.s1 |= (int)(clamp((float)( (int)(p1.s1)        >> 16) - amd_unpack3(p2.s0), -32768.0f, 32767.0f)) << 16;\n"
			"  r.s2  = (int)(clamp((float)(((int)(p1.s2) << 16) >> 16) - amd_unpack0(p2.s1), -32768.0f, 32767.0f)) & 0x0000ffff;\n"
			"  r.s2 |= (int)(clamp((float)( (int)(p1.s2)        >> 16) - amd_unpack1(p2.s1), -32768.0f, 32767.0f)) << 16;\n"
			"  r.s3  = (int)(clamp((float)(((int)(p1.s3) << 16) >> 16) - amd_unpack2(p2.s1), -32768.0f, 32767.0f)) & 0x0000ffff;\n"
			"  r.s3 |= (int)(clamp((float)( (int)(p1.s3)        >> 16) - amd_unpack3(p2.s1), -32768.0f, 32767.0f)) << 16;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Mul_S16_S16U8_Wrap_Trunc(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		vx_float32 scale = node->paramList[3]->u.scalar.u.f;
		if (HafCpu_Mul_S16_S16U8_Wrap_Trunc(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, (vx_int16 *)iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes, scale)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN_S(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8, VX_TYPE_FLOAT32);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, S16x8 p1, U8x8 p2, float p3)\n"
			"{\n"
			"  S16x8 r;\n"
			"  r.s0  = (((int)(p3 * (float)(((int)(p1.s0) << 16) >> 16) * amd_unpack0(p2.s0))) & 0x0000ffff)      ;\n"
			"  r.s0 |= (((int)(p3 * (float)( (int)(p1.s0)        >> 16) * amd_unpack1(p2.s0)))             ) << 16;\n"
			"  r.s1  = (((int)(p3 * (float)(((int)(p1.s1) << 16) >> 16) * amd_unpack2(p2.s0))) & 0x0000ffff)      ;\n"
			"  r.s1 |= (((int)(p3 * (float)( (int)(p1.s1)        >> 16) * amd_unpack3(p2.s0)))             ) << 16;\n"
			"  r.s2  = (((int)(p3 * (float)(((int)(p1.s2) << 16) >> 16) * amd_unpack0(p2.s1))) & 0x0000ffff)      ;\n"
			"  r.s2 |= (((int)(p3 * (float)( (int)(p1.s2)        >> 16) * amd_unpack1(p2.s1)))             ) << 16;\n"
			"  r.s3  = (((int)(p3 * (float)(((int)(p1.s3) << 16) >> 16) * amd_unpack2(p2.s1))) & 0x0000ffff)      ;\n"
			"  r.s3 |= (((int)(p3 * (float)( (int)(p1.s3)        >> 16) * amd_unpack3(p2.s1)))             ) << 16;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Mul_S16_S16U8_Wrap_Round(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		vx_float32 scale = node->paramList[3]->u.scalar.u.f;
		if (HafCpu_Mul_S16_S16U8_Wrap_Round(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, (vx_int16 *)iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes, scale)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN_S(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8, VX_TYPE_FLOAT32);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, S16x8 p1, U8x8 p2, float p3)\n"
			"{\n"
			"  S16x8 r;\n"
			"  r.s0  = (((int)convert_short_rte(p3 * (float)(((int)(p1.s0) << 16) >> 16) * amd_unpack0(p2.s0))) & 0x0000ffff)      ;\n"
			"  r.s0 |= (((int)convert_short_rte(p3 * (float)( (int)(p1.s0) >> 16)        * amd_unpack1(p2.s0)))             ) << 16;\n"
			"  r.s1  = (((int)convert_short_rte(p3 * (float)(((int)(p1.s1) << 16) >> 16) * amd_unpack2(p2.s0))) & 0x0000ffff)      ;\n"
			"  r.s1 |= (((int)convert_short_rte(p3 * (float)( (int)(p1.s1) >> 16)        * amd_unpack3(p2.s0)))             ) << 16;\n"
			"  r.s2  = (((int)convert_short_rte(p3 * (float)(((int)(p1.s2) << 16) >> 16) * amd_unpack0(p2.s1))) & 0x0000ffff)      ;\n"
			"  r.s2 |= (((int)convert_short_rte(p3 * (float)( (int)(p1.s2) >> 16)        * amd_unpack1(p2.s1)))             ) << 16;\n"
			"  r.s3  = (((int)convert_short_rte(p3 * (float)(((int)(p1.s3) << 16) >> 16) * amd_unpack2(p2.s1))) & 0x0000ffff)      ;\n"
			"  r.s3 |= (((int)convert_short_rte(p3 * (float)( (int)(p1.s3) >> 16)        * amd_unpack3(p2.s1)))             ) << 16;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Mul_S16_S16U8_Sat_Trunc(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		vx_float32 scale = node->paramList[3]->u.scalar.u.f;
		if (HafCpu_Mul_S16_S16U8_Sat_Trunc(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, (vx_int16 *)iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes, scale)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN_S(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8, VX_TYPE_FLOAT32);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, S16x8 p1, U8x8 p2, float p3)\n"
			"{\n"
			"  S16x8 r;\n"
			"  float f;\n"
			"  f = clamp(p3 * (float)(((int)(p1.s0) << 16) >> 16) * amd_unpack0(p2.s0), -32768.0f, 32767.0f); r.s0  = ((int)(f) & 0x0000ffff)      ;\n"
			"  f = clamp(p3 * (float)( (int)(p1.s0)        >> 16) * amd_unpack1(p2.s0), -32768.0f, 32767.0f); r.s0 |= ((int)(f)             ) << 16;\n"
			"  f = clamp(p3 * (float)(((int)(p1.s1) << 16) >> 16) * amd_unpack2(p2.s0), -32768.0f, 32767.0f); r.s1  = ((int)(f) & 0x0000ffff)      ;\n"
			"  f = clamp(p3 * (float)( (int)(p1.s1)        >> 16) * amd_unpack3(p2.s0), -32768.0f, 32767.0f); r.s1 |= ((int)(f)             ) << 16;\n"
			"  f = clamp(p3 * (float)(((int)(p1.s2) << 16) >> 16) * amd_unpack0(p2.s1), -32768.0f, 32767.0f); r.s2  = ((int)(f) & 0x0000ffff)      ;\n"
			"  f = clamp(p3 * (float)( (int)(p1.s2)        >> 16) * amd_unpack1(p2.s1), -32768.0f, 32767.0f); r.s2 |= ((int)(f)             ) << 16;\n"
			"  f = clamp(p3 * (float)(((int)(p1.s3) << 16) >> 16) * amd_unpack2(p2.s1), -32768.0f, 32767.0f); r.s3  = ((int)(f) & 0x0000ffff)      ;\n"
			"  f = clamp(p3 * (float)( (int)(p1.s3)        >> 16) * amd_unpack3(p2.s1), -32768.0f, 32767.0f); r.s3 |= ((int)(f)             ) << 16;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Mul_S16_S16U8_Sat_Round(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		vx_float32 scale = node->paramList[3]->u.scalar.u.f;
		if (HafCpu_Mul_S16_S16U8_Sat_Round(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, (vx_int16 *)iImg0->buffer, iImg0->u.img.stride_in_bytes, iImg1->buffer, iImg1->u.img.stride_in_bytes, scale)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN_S(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8, VX_TYPE_FLOAT32);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, S16x8 p1, U8x8 p2, float p3)\n"
			"{\n"
			"  S16x8 r;\n"
			"  r.s0 =  (((int)(convert_short_sat_rte(p3 * (float)(((int)(p1.s0) << 16) >> 16) * amd_unpack0(p2.s0)))) & 0x0000ffff)      ;\n"
			"  r.s0 |= (((int)(convert_short_sat_rte(p3 * (float)((int)(p1.s0)  >> 16)	       * amd_unpack1(p2.s0))))             ) << 16;\n"
			"  r.s1  = (((int)(convert_short_sat_rte(p3 * (float)(((int)(p1.s1) << 16) >> 16) * amd_unpack2(p2.s0)))) & 0x0000ffff)      ;\n"
			"  r.s1 |= (((int)(convert_short_sat_rte(p3 * (float)((int)(p1.s1)  >> 16)        * amd_unpack3(p2.s0)))))              << 16;\n"
			"  r.s2  = (((int)(convert_short_sat_rte(p3 * (float)(((int)(p1.s2) << 16) >> 16) * amd_unpack0(p2.s1)))) & 0x0000ffff)      ;\n"
			"  r.s2 |= (((int)(convert_short_sat_rte(p3 * (float)((int)(p1.s2)  >> 16)        * amd_unpack1(p2.s1)))))              << 16;\n"
			"  r.s3  = (((int)(convert_short_sat_rte(p3 * (float)(((int)(p1.s3) << 16) >> 16) * amd_unpack2(p2.s1)))) & 0x0000ffff)      ;\n"
			"  r.s3 |= (((int)(convert_short_sat_rte(p3 * (float)((int)(p1.s3)  >> 16)        * amd_unpack3(p2.s1)))))              << 16;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_AccumulateSquared_S16_S16U8_Sat(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		vx_uint32 shift = node->paramList[2]->u.scalar.u.u;
		if (HafCpu_AccumulateSquared_S16_S16U8_Sat(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes, shift)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[0]->u.img.width;
		vx_uint32 height = node->paramList[0]->u.img.height;
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_S16 || node->paramList[1]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != node->paramList[1]->u.img.width || height != node->paramList[1]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (node->paramList[2]->u.scalar.type != VX_TYPE_UINT32)
			return VX_ERROR_INVALID_TYPE;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, U8x8 p2, uint p3)\n"
			"{\n"
			"  S16x8 p1 = *p0;\n"
			"  S16x8 r; int i;\n"
			"  i = (p2.s0      ) & 255; i *= i; i >>= p3; i += (p1.s0 & 0xffff); i = clamp(i, -32768, 32767); r.s0  = i & 0xffff;\n"
			"  i = (p2.s0 >>  8) & 255; i *= i; i >>= p3; i += (p1.s0 >>    16); i = clamp(i, -32768, 32767); r.s0 |= i <<    16;\n"
			"  i = (p2.s0 >> 16) & 255; i *= i; i >>= p3; i += (p1.s1 & 0xffff); i = clamp(i, -32768, 32767); r.s1  = i & 0xffff;\n"
			"  i = (p2.s0 >> 24) & 255; i *= i; i >>= p3; i += (p1.s1 >>    16); i = clamp(i, -32768, 32767); r.s1 |= i <<    16;\n"
			"  i = (p2.s1      ) & 255; i *= i; i >>= p3; i += (p1.s2 & 0xffff); i = clamp(i, -32768, 32767); r.s2  = i & 0xffff;\n"
			"  i = (p2.s1 >>  8) & 255; i *= i; i >>= p3; i += (p1.s2 >>    16); i = clamp(i, -32768, 32767); r.s2 |= i <<    16;\n"
			"  i = (p2.s1 >> 16) & 255; i *= i; i >>= p3; i += (p1.s3 & 0xffff); i = clamp(i, -32768, 32767); r.s3  = i & 0xffff;\n"
			"  i = (p2.s1 >> 24) & 255; i *= i; i >>= p3; i += (p1.s3 >>    16); i = clamp(i, -32768, 32767); r.s3 |= i <<    16;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[0];
		AgoData * inp2 = node->paramList[1];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Sub_S16_U8S16_Wrap(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Sub_S16_U8S16_Wrap(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, (vx_int16 *)iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8, VX_DF_IMAGE_S16);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, U8x8 p1, S16x8 p2)\n"
			"{\n"
			"  S16x8 r;\n"
			"  r.s0  = (( p1.s0        & 0x000000ff) - (((int)(p2.s0) << 16) >> 16)) & 0x0000ffff;\n"
			"  r.s0 |= (((p1.s0 <<  8) & 0x00ff0000) - (       p2.s0  & 0xffff0000));\n"
			"  r.s1  = (((p1.s0 >> 16) & 0x000000ff) - (((int)(p2.s1) << 16) >> 16)) & 0x0000ffff;\n"
			"  r.s1 |= (((p1.s0 >>  8) & 0x00ff0000) - (       p2.s1  & 0xffff0000));\n"
			"  r.s2  = (( p1.s1        & 0x000000ff) - (((int)(p2.s2) << 16) >> 16)) & 0x0000ffff;\n"
			"  r.s2 |= (((p1.s1 <<  8) & 0x00ff0000) - (       p2.s2  & 0xffff0000));\n"
			"  r.s3  = (((p1.s1 >> 16) & 0x000000ff) - (((int)(p2.s3) << 16) >> 16)) & 0x0000ffff;\n"
			"  r.s3 |= (((p1.s1 >>  8) & 0x00ff0000) - (       p2.s3  & 0xffff0000));\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Sub_S16_U8S16_Sat(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Sub_S16_U8S16_Sat(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, iImg0->buffer, iImg0->u.img.stride_in_bytes, (vx_int16 *)iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8, VX_DF_IMAGE_S16);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		//NOTE: Check line 6489 from line 1014 in ago_haf_gpu_elemwise.cpp
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, U8x8 p1, S16x8 p2)\n"
			"{\n"
			"  S16x8 r;\n"
			"  r.s0  = (int)(clamp(amd_unpack0(p1.s0) - (float)(((int)(p2.s0) << 16) >> 16), -32768.0f, 32767.0f)) & 0x0000ffff;\n"
			"  r.s0 |= (int)(clamp(amd_unpack1(p1.s0) - (float)( (int)(p2.s0)        >> 16), -32768.0f, 32767.0f)) << 16;\n"
			"  r.s1  = (int)(clamp(amd_unpack2(p1.s0) - (float)(((int)(p2.s1) << 16) >> 16), -32768.0f, 32767.0f)) & 0x0000ffff;\n"
			"  r.s1 |= (int)(clamp(amd_unpack3(p1.s0) - (float)( (int)(p2.s1)        >> 16), -32768.0f, 32767.0f)) << 16;\n"
			"  r.s2  = (int)(clamp(amd_unpack0(p1.s1) - (float)(((int)(p2.s2) << 16) >> 16), -32768.0f, 32767.0f)) & 0x0000ffff;\n"
			"  r.s2 |= (int)(clamp(amd_unpack1(p1.s1) - (float)( (int)(p2.s2)        >> 16), -32768.0f, 32767.0f)) << 16;\n"
			"  r.s3  = (int)(clamp(amd_unpack2(p1.s1) - (float)(((int)(p2.s3) << 16) >> 16), -32768.0f, 32767.0f)) & 0x0000ffff;\n"
			"  r.s3 |= (int)(clamp(amd_unpack3(p1.s1) - (float)( (int)(p2.s3)        >> 16), -32768.0f, 32767.0f)) << 16;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_AbsDiff_S16_S16S16_Sat(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_AbsDiff_S16_S16S16_Sat(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, (vx_int16 *)iImg0->buffer, iImg0->u.img.stride_in_bytes, (vx_int16 *)iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_AbsDiff_S16_S16S16_Sat(node->opencl_code);
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		char item[512];
		if (iImg1->u.img.isUniform && !iImg0->u.img.isUniform) {
			// avoid having to read constant uniform image for AbsDiff (users might do this for Abs operation)
			node->opencl_param_discard_mask = (1 << 2);
			sprintf(item, "#define %s(p0,p1) AbsDiff_S16_S16S16_Sat(p0,p1,(S16x8)(%d))\n", node->opencl_name, (int)iImg1->u.img.uniform[0]);
			node->opencl_code += item;
		}
		else if(iImg0->u.img.isUniform && !iImg1->u.img.isUniform) {
			// avoid having to read constant uniform image for AbsDiff (users might do this for Abs operation)
			node->opencl_param_discard_mask = (1 << 1);
			sprintf(item, "#define %s(p0,p2) AbsDiff_S16_S16S16_Sat(p0,p2,(S16x8)(%d))\n", node->opencl_name, (int)iImg0->u.img.uniform[0]);
			node->opencl_code += item;
		}
		else {
			sprintf(item, "#define %s(p0,p1,p2) AbsDiff_S16_S16S16_Sat(p0,p1,p2)\n", node->opencl_name);
			node->opencl_code += item;
		}
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Add_S16_S16S16_Wrap(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Add_S16_S16S16_Wrap(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, (vx_int16 *)iImg0->buffer, iImg0->u.img.stride_in_bytes, (vx_int16 *)iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, S16x8 p1, S16x8 p2)\n"
			"{\n"
			"  S16x8 r;\n"
			"  r.s0  = (p1.s0 +  p2.s0              ) & 0x0000ffff;\n"
			"  r.s0 |= (p1.s0 + (p2.s0 & 0xffff0000)) & 0xffff0000;\n"
			"  r.s1  = (p1.s1 +  p2.s1              ) & 0x0000ffff;\n"
			"  r.s1 |= (p1.s1 + (p2.s1 & 0xffff0000)) & 0xffff0000;\n"
			"  r.s2  = (p1.s2 +  p2.s2              ) & 0x0000ffff;\n"
			"  r.s2 |= (p1.s2 + (p2.s2 & 0xffff0000)) & 0xffff0000;\n"
			"  r.s3  = (p1.s3 +  p2.s3              ) & 0x0000ffff;\n"
			"  r.s3 |= (p1.s3 + (p2.s3 & 0xffff0000)) & 0xffff0000;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Add_S16_S16S16_Sat(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Add_S16_S16S16_Sat(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, (vx_int16 *)iImg0->buffer, iImg0->u.img.stride_in_bytes, (vx_int16 *)iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, S16x8 p1, S16x8 p2)\n"
			"{\n"
			"  S16x8 r;\n"
			"  r.s0  = clamp((((int)(p1.s0) << 16) >> 16) + (((int)(p2.s0) << 16) >> 16), -32768, 32767) & 0x0000ffff;\n"
			"  r.s0 |= clamp(( (int)(p1.s0)        >> 16) + ( (int)(p2.s0)        >> 16), -32768, 32767) << 16;\n"
			"  r.s1  = clamp((((int)(p1.s1) << 16) >> 16) + (((int)(p2.s1) << 16) >> 16), -32768, 32767) & 0x0000ffff;\n"
			"  r.s1 |= clamp(( (int)(p1.s1)        >> 16) + ( (int)(p2.s1)        >> 16), -32768, 32767) << 16;\n"
			"  r.s2  = clamp((((int)(p1.s2) << 16) >> 16) + (((int)(p2.s2) << 16) >> 16), -32768, 32767) & 0x0000ffff;\n"
			"  r.s2 |= clamp(( (int)(p1.s2)        >> 16) + ( (int)(p2.s2)        >> 16), -32768, 32767) << 16;\n"
			"  r.s3  = clamp((((int)(p1.s3) << 16) >> 16) + (((int)(p2.s3) << 16) >> 16), -32768, 32767) & 0x0000ffff;\n"
			"  r.s3 |= clamp(( (int)(p1.s3)        >> 16) + ( (int)(p2.s3)        >> 16), -32768, 32767) << 16;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Sub_S16_S16S16_Wrap(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Sub_S16_S16S16_Wrap(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, (vx_int16 *)iImg0->buffer, iImg0->u.img.stride_in_bytes, (vx_int16 *)iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, S16x8 p1, S16x8 p2)\n"
			"{\n"
			"  S16x8 r;\n"
			"  r.s0  = (p1.s0 -  p2.s0              ) & 0x0000ffff;\n"
			"  r.s0 |= (p1.s0 - (p2.s0 & 0xffff0000)) & 0xffff0000;\n"
			"  r.s1  = (p1.s1 -  p2.s1              ) & 0x0000ffff;\n"
			"  r.s1 |= (p1.s1 - (p2.s1 & 0xffff0000)) & 0xffff0000;\n"
			"  r.s2  = (p1.s2 -  p2.s2              ) & 0x0000ffff;\n"
			"  r.s2 |= (p1.s2 - (p2.s2 & 0xffff0000)) & 0xffff0000;\n"
			"  r.s3  = (p1.s3 -  p2.s3              ) & 0x0000ffff;\n"
			"  r.s3 |= (p1.s3 - (p2.s3 & 0xffff0000)) & 0xffff0000;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Sub_S16_S16S16_Sat(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Sub_S16_S16S16_Sat(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, (vx_int16 *)iImg0->buffer, iImg0->u.img.stride_in_bytes, (vx_int16 *)iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, S16x8 p1, S16x8 p2)\n"
			"{\n"
			"  S16x8 r;\n"
			"  r.s0  = clamp((((int)(p1.s0) << 16) >> 16) - (((int)(p2.s0) << 16) >> 16), -32768, 32767) & 0x0000ffff;\n"
			"  r.s0 |= clamp(( (int)(p1.s0)        >> 16) - ( (int)(p2.s0)        >> 16), -32768, 32767) << 16;\n"
			"  r.s1  = clamp((((int)(p1.s1) << 16) >> 16) - (((int)(p2.s1) << 16) >> 16), -32768, 32767) & 0x0000ffff;\n"
			"  r.s1 |= clamp(( (int)(p1.s1)        >> 16) - ( (int)(p2.s1)        >> 16), -32768, 32767) << 16;\n"
			"  r.s2  = clamp((((int)(p1.s2) << 16) >> 16) - (((int)(p2.s2) << 16) >> 16), -32768, 32767) & 0x0000ffff;\n"
			"  r.s2 |= clamp(( (int)(p1.s2)        >> 16) - ( (int)(p2.s2)        >> 16), -32768, 32767) << 16;\n"
			"  r.s3  = clamp((((int)(p1.s3) << 16) >> 16) - (((int)(p2.s3) << 16) >> 16), -32768, 32767) & 0x0000ffff;\n"
			"  r.s3 |= clamp(( (int)(p1.s3)        >> 16) - ( (int)(p2.s3)        >> 16), -32768, 32767) << 16;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Mul_S16_S16S16_Wrap_Trunc(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		vx_float32 scale = node->paramList[3]->u.scalar.u.f;
		if (HafCpu_Mul_S16_S16S16_Wrap_Trunc(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, (vx_int16 *)iImg0->buffer, iImg0->u.img.stride_in_bytes, (vx_int16 *)iImg1->buffer, iImg1->u.img.stride_in_bytes, scale)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN_S(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16, VX_TYPE_FLOAT32);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, S16x8 p1, S16x8 p2, float p3)\n"
			"{\n"
			"  S16x8 r;\n"
			"  r.s0  = ((int)(p3 * (double)((((int)(p1.s0)) << 16) >> 16) * (double)((((int)(p2.s0)) << 16) >> 16))) & 0x0000ffff;\n"
			"  r.s0 |= ((int)(p3 * (double)(( (int)(p1.s0)) >> 16)        * (double)(( (int)(p2.s0)) >> 16)))		  << 16;\n"
			"  r.s1  = ((int)(p3 * (double)((((int)(p1.s1)) << 16) >> 16) * (double)((((int)(p2.s1)) << 16) >> 16))) & 0x0000ffff;\n"
			"  r.s1 |= ((int)(p3 * (double)(( (int)(p1.s1)) >> 16)        * (double)(( (int)(p2.s1)) >> 16)))		  << 16;\n"
			"  r.s2  = ((int)(p3 * (double)((((int)(p1.s2)) << 16) >> 16) * (double)((((int)(p2.s2)) << 16) >> 16))) & 0x0000ffff;\n"
			"  r.s2 |= ((int)(p3 * (double)(( (int)(p1.s2)) >> 16)        * (double)(( (int)(p2.s2)) >> 16)))		  << 16;\n"
			"  r.s3  = ((int)(p3 * (double)((((int)(p1.s3)) << 16) >> 16) * (double)((((int)(p2.s3)) << 16) >> 16))) & 0x0000ffff;\n"
			"  r.s3 |= ((int)(p3 * (double)(( (int)(p1.s3)) >> 16)        * (double)(( (int)(p2.s3)) >> 16)))		  << 16;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Mul_S16_S16S16_Wrap_Round(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		vx_float32 scale = node->paramList[3]->u.scalar.u.f;
		if (HafCpu_Mul_S16_S16S16_Wrap_Round(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, (vx_int16 *)iImg0->buffer, iImg0->u.img.stride_in_bytes, (vx_int16 *)iImg1->buffer, iImg1->u.img.stride_in_bytes, scale)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN_S(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16, VX_TYPE_FLOAT32);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, S16x8 p1, S16x8 p2, float p3)\n"
			"{\n"
			"  S16x8 r;\n"
			"  r.s0  = ((int)convert_short_rte(p3 * (float)((((int)(p1.s0)) << 16) >> 16) * (float)((((int)(p2.s0)) << 16) >> 16)) & 0x0000ffff)      ;\n"
			"  r.s0 |= ((int)convert_short_rte(p3 * (float)(((int)(p1.s0))  >> 16)        * (float)(((int)(p2.s0))  >> 16))                    ) << 16;\n"
			"  r.s1  = ((int)convert_short_rte(p3 * (float)((((int)(p1.s1)) << 16) >> 16) * (float)((((int)(p2.s1)) << 16) >> 16)) & 0x0000ffff);\n"
			"  r.s1 |= ((int)convert_short_rte(p3 * (float)(((int)(p1.s1))  >> 16)        * (float)(((int)(p2.s1))  >> 16))                    ) << 16;\n"
			"  r.s2  = ((int)convert_short_rte(p3 * (float)((((int)(p1.s2)) << 16) >> 16) * (float)((((int)(p2.s2)) << 16) >> 16)) & 0x0000ffff)      ;\n"
			"  r.s2 |= ((int)convert_short_rte(p3 * (float)(((int)(p1.s2))  >> 16)        * (float)(((int)(p2.s2))  >> 16))                    ) << 16;\n"
			"  r.s3  = ((int)convert_short_rte(p3 * (float)((((int)(p1.s3)) << 16) >> 16) * (float)((((int)(p2.s3)) << 16) >> 16)) & 0x0000ffff)      ;\n"
			"  r.s3 |= ((int)convert_short_rte(p3 * (float)(((int)(p1.s3))  >> 16)        * (float)(((int)(p2.s3))  >> 16))                    ) << 16;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Mul_S16_S16S16_Sat_Trunc(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		vx_float32 scale = node->paramList[3]->u.scalar.u.f;
		if (HafCpu_Mul_S16_S16S16_Sat_Trunc(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, (vx_int16 *)iImg0->buffer, iImg0->u.img.stride_in_bytes, (vx_int16 *)iImg1->buffer, iImg1->u.img.stride_in_bytes, scale)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN_S(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16, VX_TYPE_FLOAT32);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, S16x8 p1, S16x8 p2, float p3)\n"
			"{\n"
			"  S16x8 r;\n"
			"  r.s0  = (((int)clamp((p3 * (float)((((int)(p1.s0)) << 16) >> 16) * (float)((((int)(p2.s0)) << 16) >> 16)), -32768.0f, 32767.0f)) & 0x0000ffff)      ;\n"
			"  r.s0 |= (((int)clamp((p3 * (float)(( (int)(p1.s0))        >> 16) * (float)(( (int)(p2.s0))        >> 16)), -32768.0f, 32767.0f))             ) << 16;\n"
			"  r.s1  = (((int)clamp((p3 * (float)((((int)(p1.s1)) << 16) >> 16) * (float)((((int)(p2.s1)) << 16) >> 16)), -32768.0f, 32767.0f)) & 0x0000ffff)      ;\n"
			"  r.s1 |= (((int)clamp((p3 * (float)(( (int)(p1.s1))        >> 16) * (float)(( (int)(p2.s1))        >> 16)), -32768.0f, 32767.0f))             ) << 16;\n"
			"  r.s2  = (((int)clamp((p3 * (float)((((int)(p1.s2)) << 16) >> 16) * (float)((((int)(p2.s2)) << 16) >> 16)), -32768.0f, 32767.0f)) & 0x0000ffff)      ;\n"
			"  r.s2 |= (((int)clamp((p3 * (float)(( (int)(p1.s2))        >> 16) * (float)(( (int)(p2.s2))        >> 16)), -32768.0f, 32767.0f))             ) << 16;\n"
			"  r.s3  = (((int)clamp((p3 * (float)((((int)(p1.s3)) << 16) >> 16) * (float)((((int)(p2.s3)) << 16) >> 16)), -32768.0f, 32767.0f)) & 0x0000ffff)      ;\n"
			"  r.s3 |= (((int)clamp((p3 * (float)(( (int)(p1.s3))        >> 16) * (float)(( (int)(p2.s3))        >> 16)), -32768.0f, 32767.0f))             ) << 16;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Mul_S16_S16S16_Sat_Round(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		vx_float32 scale = node->paramList[3]->u.scalar.u.f;
		if (HafCpu_Mul_S16_S16S16_Sat_Round(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, (vx_int16 *)iImg0->buffer, iImg0->u.img.stride_in_bytes, (vx_int16 *)iImg1->buffer, iImg1->u.img.stride_in_bytes, scale)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN_S(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16, VX_TYPE_FLOAT32);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, S16x8 p1, S16x8 p2, float p3)\n"
			"{\n"
			"  S16x8 r;\n"
			"  r.s0  = (((int)convert_short_sat_rte(p3 * (float)((((int)(p1.s0)) << 16) >> 16) * (float)((((int)(p2.s0)) << 16) >> 16))) & 0x0000ffff)      ;\n"
			"  r.s0 |= (((int)convert_short_sat_rte(p3 * (float)(((int)(p1.s0))  >> 16)        * (float)(( (int)(p2.s0)) >> 16)      )))               << 16;\n"
			"  r.s1  = (((int)convert_short_sat_rte(p3 * (float)((((int)(p1.s1)) << 16) >> 16) * (float)((((int)(p2.s1)) << 16) >> 16))) & 0x0000ffff)      ;\n"
			"  r.s1 |= (((int)convert_short_sat_rte(p3 * (float)(((int)(p1.s1))  >> 16)        * (float)(( (int)(p2.s1)) >> 16)      )))               << 16;\n"
			"  r.s2  = (((int)convert_short_sat_rte(p3 * (float)((((int)(p1.s2)) << 16) >> 16) * (float)((((int)(p2.s2)) << 16) >> 16))) & 0x0000ffff)      ;\n"
			"  r.s2 |= (((int)convert_short_sat_rte(p3 * (float)(((int)(p1.s2))  >> 16)        * (float)(( (int)(p2.s2)) >> 16)      )))               << 16;\n"
			"  r.s3  = (((int)convert_short_sat_rte(p3 * (float)((((int)(p1.s3)) << 16) >> 16) * (float)((((int)(p2.s3)) << 16) >> 16))) & 0x0000ffff)      ;\n"
			"  r.s3 |= (((int)convert_short_sat_rte(p3 * (float)(((int)(p1.s3))  >> 16)        * (float)(( (int)(p2.s3)) >> 16)      )))               << 16;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Magnitude_S16_S16S16(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Magnitude_S16_S16S16(oImg->u.img.width, oImg->u.img.height, (vx_int16 *)oImg->buffer, oImg->u.img.stride_in_bytes, (vx_int16 *)iImg0->buffer, iImg0->u.img.stride_in_bytes, (vx_int16 *)iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[4096];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (S16x8 * p0, S16x8 p1, S16x8 p2)\n"
			"{\n"
			"  S16x8 r;\n"
			"  float2 f;\n"
			"  f.s0 = (float)((((int)(p1.s0)) << 16) >> 16); f.s1 = (float)((((int)(p2.s0)) << 16) >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s0  = (uint)(f.s0);\n"
			"  f.s0 = (float)(( (int)(p1.s0))        >> 16); f.s1 = (float)(( (int)(p2.s0))        >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s0 |= (uint)(f.s0) << 16;\n"
			"  f.s0 = (float)((((int)(p1.s1)) << 16) >> 16); f.s1 = (float)((((int)(p2.s1)) << 16) >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s1  = (uint)(f.s0);\n"
			"  f.s0 = (float)(( (int)(p1.s1))        >> 16); f.s1 = (float)(( (int)(p2.s1))        >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s1 |= (uint)(f.s0) << 16;\n"
			"  f.s0 = (float)((((int)(p1.s2)) << 16) >> 16); f.s1 = (float)((((int)(p2.s2)) << 16) >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s2  = (uint)(f.s0);\n"
			"  f.s0 = (float)(( (int)(p1.s2))        >> 16); f.s1 = (float)(( (int)(p2.s2))        >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s2 |= (uint)(f.s0) << 16;\n"
			"  f.s0 = (float)((((int)(p1.s3)) << 16) >> 16); f.s1 = (float)((((int)(p2.s3)) << 16) >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s3  = (uint)(f.s0);\n"
			"  f.s0 = (float)(( (int)(p1.s3))        >> 16); f.s1 = (float)(( (int)(p2.s3))        >> 16); f.s0 *= f.s0; f.s0 = mad(f.s1, f.s1, f.s0); f.s0 = native_sqrt(f.s0); f.s0 = min(f.s0 + 0.5f, 32767.0f); r.s3 |= (uint)(f.s0) << 16;\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Phase_U8_S16S16(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg0 = node->paramList[1];
		AgoData * iImg1 = node->paramList[2];
		if (HafCpu_Phase_U8_S16S16(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, (vx_int16 *)iImg0->buffer, iImg0->u.img.stride_in_bytes, (vx_int16 *)iImg1->buffer, iImg1->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, S16x8 p1, S16x8 p2)\n"
			"{\n"
			"  U8x8 r;\n"
			"  float2 f; float4 p4;\n"
			"  f.s0 = (float)((((int)(p1.s0)) << 16) >> 16); f.s1 = (float)((((int)(p2.s0)) << 16) >> 16); p4.s0 = atan2pi(f.s1, f.s0); p4.s0 += (p4.s0 < 0.0) ? 2.0f : 0.0; p4.s0 *= 128.0f;\n"
			"  f.s0 = (float)(( (int)(p1.s0))        >> 16); f.s1 = (float)(( (int)(p2.s0))        >> 16); p4.s1 = atan2pi(f.s1, f.s0); p4.s1 += (p4.s1 < 0.0) ? 2.0f : 0.0; p4.s1 *= 128.0f;\n"
			"  f.s0 = (float)((((int)(p1.s1)) << 16) >> 16); f.s1 = (float)((((int)(p2.s1)) << 16) >> 16); p4.s2 = atan2pi(f.s1, f.s0); p4.s2 += (p4.s2 < 0.0) ? 2.0f : 0.0; p4.s2 *= 128.0f;\n"
			"  f.s0 = (float)(( (int)(p1.s1))        >> 16); f.s1 = (float)(( (int)(p2.s1))        >> 16); p4.s3 = atan2pi(f.s1, f.s0); p4.s3 += (p4.s3 < 0.0) ? 2.0f : 0.0; p4.s3 *= 128.0f;\n"
			"  p4 = select(p4, (float4) 0.0f, p4 > 255.5f);\n"
			"  r.s0 = amd_pack(p4);\n"
			"  f.s0 = (float)((((int)(p1.s2)) << 16) >> 16); f.s1 = (float)((((int)(p2.s2)) << 16) >> 16); p4.s0 = atan2pi(f.s1, f.s0); p4.s0 += (p4.s0 < 0.0) ? 2.0f : 0.0; p4.s0 *= 128.0f;\n"
			"  f.s0 = (float)(( (int)(p1.s2))        >> 16); f.s1 = (float)(( (int)(p2.s2))        >> 16); p4.s1 = atan2pi(f.s1, f.s0); p4.s1 += (p4.s1 < 0.0) ? 2.0f : 0.0; p4.s1 *= 128.0f;\n"
			"  f.s0 = (float)((((int)(p1.s3)) << 16) >> 16); f.s1 = (float)((((int)(p2.s3)) << 16) >> 16); p4.s2 = atan2pi(f.s1, f.s0); p4.s2 += (p4.s2 < 0.0) ? 2.0f : 0.0; p4.s2 *= 128.0f;\n"
			"  f.s0 = (float)(( (int)(p1.s3))        >> 16); f.s1 = (float)(( (int)(p2.s3))        >> 16); p4.s3 = atan2pi(f.s1, f.s0); p4.s3 += (p4.s3 < 0.0) ? 2.0f : 0.0; p4.s3 *= 128.0f;\n"
			"  p4 = select(p4, (float4) 0.0f, p4 > 255.5f);\n"
			"  r.s1 = amd_pack(p4);\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_ChannelCopy_U8_U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ChannelCopy_U8_U8(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 p1)\n"
			"{\n"
			"  *p0 = p1;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ChannelCopy_U8_U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ChannelCopy_U8_U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U1x8 p1)\n"
			"{\n"
			"  Convert_U8_U1(p0, p1);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ChannelCopy_U1_U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ChannelCopy_U1_U8(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U8x8 p1)\n"
			"{\n"
			"  Convert_U1_U8(p0, p1);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ChannelCopy_U1_U1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ChannelCopy_U1_U1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U1x8 * p0, U1x8 p1)\n"
			"{\n"
			"  *p0 = p1;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ChannelExtract_U8_U16_Pos0(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ChannelExtract_U8_U16_Pos0(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = VX_SUCCESS;
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_U16 && node->paramList[1]->u.img.format != VX_DF_IMAGE_YUYV && node->paramList[1]->u.img.format != VX_DF_IMAGE_UYVY)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U16x8 p1)\n"
			"{\n"
			"  U8x8 r;\n"
			"  r.s0 = amd_pack((float4)(amd_unpack0(p1.s0), amd_unpack2(p1.s0), amd_unpack0(p1.s1), amd_unpack2(p1.s1)));\n"
			"  r.s1 = amd_pack((float4)(amd_unpack0(p1.s2), amd_unpack2(p1.s2), amd_unpack0(p1.s3), amd_unpack2(p1.s3)));\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ChannelExtract_U8_U16_Pos1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ChannelExtract_U8_U16_Pos1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = VX_SUCCESS;
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_U16 && node->paramList[1]->u.img.format != VX_DF_IMAGE_YUYV && node->paramList[1]->u.img.format != VX_DF_IMAGE_UYVY)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U16x8 p1)\n"
			"{\n"
			"  U8x8 r;\n"
			"  r.s0 = amd_pack((float4)(amd_unpack1(p1.s0), amd_unpack3(p1.s0), amd_unpack1(p1.s1), amd_unpack3(p1.s1)));\n"
			"  r.s1 = amd_pack((float4)(amd_unpack1(p1.s2), amd_unpack3(p1.s2), amd_unpack1(p1.s3), amd_unpack3(p1.s3)));\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ChannelExtract_U8_U24_Pos0(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ChannelExtract_U8_U24_Pos0(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_RGB);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_ChannelExtract_U8_U24_Pos0(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"#define %s ChannelExtract_U8_U24_Pos0\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ChannelExtract_U8_U24_Pos1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ChannelExtract_U8_U24_Pos1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_RGB);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_ChannelExtract_U8_U24_Pos1(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"#define %s ChannelExtract_U8_U24_Pos1\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ChannelExtract_U8_U24_Pos2(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ChannelExtract_U8_U24_Pos2(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_RGB);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_ChannelExtract_U8_U24_Pos2(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"#define %s ChannelExtract_U8_U24_Pos2\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ChannelExtract_U8_U32_Pos0(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ChannelExtract_U8_U32_Pos0(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = VX_SUCCESS;
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		vx_df_image format = node->paramList[1]->u.img.format;
		if (format != VX_DF_IMAGE_RGBX && format != VX_DF_IMAGE_UYVY)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width >> (format == VX_DF_IMAGE_RGBX ? 0 : 1);
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		AgoData * iImg = node->paramList[1];
		if (iImg->u.img.format == VX_DF_IMAGE_RGBX){
			node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
			agoCodeGenOpenCL_ChannelExtract_U8_U32_Pos0(node->opencl_code);
			char textBuffer[2048];
			sprintf(textBuffer, OPENCL_FORMAT(
				"#define %s ChannelExtract_U8_U32_Pos0\n"
				), node->opencl_name);
			node->opencl_code += textBuffer;
		}
		else if (iImg->u.img.format == VX_DF_IMAGE_UYVY)
			status = HafGpu_ChannelExtract_U8_U32(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
		AgoData * iImg = node->paramList[1];
		if (iImg->u.img.format == VX_DF_IMAGE_RGBX)	{
			node->target_support_flags = 0
				| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL				
				| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif				
				;
		}
		else if (iImg->u.img.format == VX_DF_IMAGE_UYVY)	{
			node->target_support_flags = 0
				| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL				
				| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif				
				;
		}
        
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ChannelExtract_U8_U32_Pos1(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ChannelExtract_U8_U32_Pos1(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = VX_SUCCESS;
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		vx_df_image format = node->paramList[1]->u.img.format;
		if (format != VX_DF_IMAGE_RGBX && format != VX_DF_IMAGE_YUYV)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width >> (format == VX_DF_IMAGE_RGBX ? 0 : 1);
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		AgoData * iImg = node->paramList[1];
		if (iImg->u.img.format == VX_DF_IMAGE_RGBX){
			node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
			agoCodeGenOpenCL_ChannelExtract_U8_U32_Pos1(node->opencl_code);
			char textBuffer[2048];
			sprintf(textBuffer, OPENCL_FORMAT(
				"#define %s ChannelExtract_U8_U32_Pos1\n"
				), node->opencl_name);
			node->opencl_code += textBuffer;
		}
		else if (iImg->u.img.format == VX_DF_IMAGE_YUYV)
			status = HafGpu_ChannelExtract_U8_U32(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
		AgoData * iImg = node->paramList[1];
		if (iImg->u.img.format == VX_DF_IMAGE_RGBX)	{
			node->target_support_flags = 0
				| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL				
				| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif				
				;
		}
		else if (iImg->u.img.format == VX_DF_IMAGE_YUYV)	{
			node->target_support_flags = 0
				| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL	
				| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif				
				;
		}
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ChannelExtract_U8_U32_Pos2(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ChannelExtract_U8_U32_Pos2(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = VX_SUCCESS;
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		vx_df_image format = node->paramList[1]->u.img.format;
		if (format != VX_DF_IMAGE_RGBX && format != VX_DF_IMAGE_UYVY)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width >> (format == VX_DF_IMAGE_RGBX ? 0 : 1);
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		AgoData * iImg = node->paramList[1];
		if (iImg->u.img.format == VX_DF_IMAGE_RGBX){
			node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
			agoCodeGenOpenCL_ChannelExtract_U8_U32_Pos2(node->opencl_code);
			char textBuffer[2048];
			sprintf(textBuffer, OPENCL_FORMAT(
				"#define %s ChannelExtract_U8_U32_Pos2\n"
				), node->opencl_name);
			node->opencl_code += textBuffer;
		}
		else if (iImg->u.img.format == VX_DF_IMAGE_UYVY)
			status = HafGpu_ChannelExtract_U8_U32(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
		AgoData * iImg = node->paramList[1];
		if (iImg->u.img.format == VX_DF_IMAGE_RGBX)	{
			node->target_support_flags = 0
				| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL				
				| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif				
				;
		}
		else if (iImg->u.img.format == VX_DF_IMAGE_UYVY)	{
			node->target_support_flags = 0
				| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL				
				| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif				
				;
		}
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ChannelExtract_U8_U32_Pos3(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ChannelExtract_U8_U32_Pos3(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = VX_SUCCESS;
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		vx_df_image format = node->paramList[1]->u.img.format;
		if (format != VX_DF_IMAGE_RGBX && format != VX_DF_IMAGE_YUYV)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width >> (format == VX_DF_IMAGE_RGBX ? 0 : 1);
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		AgoData * iImg = node->paramList[1];
		if (iImg->u.img.format == VX_DF_IMAGE_RGBX){
			node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
			agoCodeGenOpenCL_ChannelExtract_U8_U32_Pos3(node->opencl_code);
			char textBuffer[2048];
			sprintf(textBuffer, OPENCL_FORMAT(
				"#define %s ChannelExtract_U8_U32_Pos3\n"
				), node->opencl_name);
			node->opencl_code += textBuffer;
		}
		else if (iImg->u.img.format == VX_DF_IMAGE_YUYV)
			status = HafGpu_ChannelExtract_U8_U32(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
		AgoData * iImg = node->paramList[1];
		if (iImg->u.img.format == VX_DF_IMAGE_RGBX)	{
			node->target_support_flags = 0
				| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL				
				| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif				
				;
		}
		else if (iImg->u.img.format == VX_DF_IMAGE_YUYV)	{
			node->target_support_flags = 0
				| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL				
				| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif				
				;
		}
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ChannelExtract_U8U8U8_U24(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg0 = node->paramList[0];
		AgoData * oImg1 = node->paramList[1];
		AgoData * oImg2 = node->paramList[2];
		AgoData * iImg = node->paramList[3];
		if (HafCpu_ChannelExtract_U8U8U8_U24(oImg0->u.img.width, oImg0->u.img.height, oImg0->buffer, oImg1->buffer, oImg2->buffer, oImg0->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_3OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_RGB);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_ChannelExtract_U8_U24_Pos0(node->opencl_code);
		agoCodeGenOpenCL_ChannelExtract_U8_U24_Pos1(node->opencl_code);
		agoCodeGenOpenCL_ChannelExtract_U8_U24_Pos2(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 * p1, U8x8 * p2, U24x8 p3)\n"
			"{\n"
			"  ChannelExtract_U8_U24_Pos0(p0, p3);\n"
			"  ChannelExtract_U8_U24_Pos1(p1, p3);\n"
			"  ChannelExtract_U8_U24_Pos2(p2, p3);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out1 = node->paramList[0];
		AgoData * out2 = node->paramList[1];
		AgoData * out3 = node->paramList[2];
		AgoData * inp = node->paramList[3];
		out1->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out1->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out1->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out1->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
		out2->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out2->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out2->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out2->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
		out3->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out3->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out3->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out3->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ChannelExtract_U8U8U8_U32(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg0 = node->paramList[0];
		AgoData * oImg1 = node->paramList[1];
		AgoData * oImg2 = node->paramList[2];
		AgoData * iImg = node->paramList[3];
		if (HafCpu_ChannelExtract_U8U8U8_U32(oImg0->u.img.width, oImg0->u.img.height, oImg0->buffer, oImg1->buffer, oImg2->buffer, oImg0->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_3OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_RGBX);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_ChannelExtract_U8_U32_Pos0(node->opencl_code);
		agoCodeGenOpenCL_ChannelExtract_U8_U32_Pos1(node->opencl_code);
		agoCodeGenOpenCL_ChannelExtract_U8_U32_Pos2(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 * p1, U8x8 * p2, U32x8 p3)\n"
			"{\n"
			"  ChannelExtract_U8_U32_Pos0(p0, p3);\n"
			"  ChannelExtract_U8_U32_Pos1(p1, p3);\n"
			"  ChannelExtract_U8_U32_Pos2(p2, p3);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out1 = node->paramList[0];
		AgoData * out2 = node->paramList[1];
		AgoData * out3 = node->paramList[2];
		AgoData * inp = node->paramList[3];
		out1->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out1->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out1->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out1->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
		out2->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out2->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out2->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out2->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
		out3->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out3->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out3->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out3->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ChannelExtract_U8U8U8U8_U32(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg0 = node->paramList[0];
		AgoData * oImg1 = node->paramList[1];
		AgoData * oImg2 = node->paramList[2];
		AgoData * oImg3 = node->paramList[3];
		AgoData * iImg = node->paramList[4];
		if (HafCpu_ChannelExtract_U8U8U8U8_U32(oImg0->u.img.width, oImg0->u.img.height, oImg0->buffer, oImg1->buffer, oImg2->buffer, oImg3->buffer, oImg0->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_4OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_RGBX);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_ChannelExtract_U8_U32_Pos0(node->opencl_code);
		agoCodeGenOpenCL_ChannelExtract_U8_U32_Pos1(node->opencl_code);
		agoCodeGenOpenCL_ChannelExtract_U8_U32_Pos2(node->opencl_code);
		agoCodeGenOpenCL_ChannelExtract_U8_U32_Pos3(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 * p1, U8x8 * p2, U8x8 * p3, U32x8 p4)\n"
			"{\n"
			"  ChannelExtract_U8_U32_Pos0(p0, p4);\n"
			"  ChannelExtract_U8_U32_Pos1(p1, p4);\n"
			"  ChannelExtract_U8_U32_Pos2(p2, p4);\n"
			"  ChannelExtract_U8_U32_Pos3(p3, p4);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out1 = node->paramList[0];
		AgoData * out2 = node->paramList[1];
		AgoData * out3 = node->paramList[2];
		AgoData * out4 = node->paramList[3];
		AgoData * inp = node->paramList[4];
		out1->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out1->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out1->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out1->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
		out2->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out2->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out2->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out2->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
		out3->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out3->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out3->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out3->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
		out4->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out4->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out4->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out4->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ChannelCombine_U16_U8U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg1 = node->paramList[1];
		AgoData * iImg2 = node->paramList[2];
		if (HafCpu_ChannelCombine_U16_U8U8(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
										   iImg1->buffer, iImg1->u.img.stride_in_bytes, iImg2->buffer, iImg2->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U16, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U16x8 * p0, U8x8 p1, U8x8 p2)\n"
			"{\n"
			"  U16x8 r;\n"
			"  r.s0 = amd_pack((float4)(amd_unpack0(p1.s0), amd_unpack0(p2.s0), amd_unpack1(p1.s0), amd_unpack1(p2.s0)));\n"
			"  r.s1 = amd_pack((float4)(amd_unpack2(p1.s0), amd_unpack2(p2.s0), amd_unpack3(p1.s0), amd_unpack3(p2.s0)));\n"
			"  r.s2 = amd_pack((float4)(amd_unpack0(p1.s1), amd_unpack0(p2.s1), amd_unpack1(p1.s1), amd_unpack1(p2.s1)));\n"
			"  r.s3 = amd_pack((float4)(amd_unpack2(p1.s1), amd_unpack2(p2.s1), amd_unpack3(p1.s1), amd_unpack3(p2.s1)));\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out  = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_ChannelCombine_U24_U8U8U8_RGB(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg1 = node->paramList[1];
		AgoData * iImg2 = node->paramList[2];
		AgoData * iImg3 = node->paramList[3];
		if (HafCpu_ChannelCombine_U24_U8U8U8_RGB(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, 
												 iImg1->buffer, iImg1->u.img.stride_in_bytes, iImg2->buffer, iImg2->u.img.stride_in_bytes, 
												 iImg3->buffer, iImg3->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_3IN(node, VX_DF_IMAGE_RGB, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U24x8 * p0, U8x8 p1, U8x8 p2, U8x8 p3)\n"
			"{\n"
			"  (*p0).s0 = amd_pack((float4)(amd_unpack0(p1.s0), amd_unpack0(p2.s0), amd_unpack0(p3.s0), amd_unpack1(p1.s0)));\n"
			"  (*p0).s1 = amd_pack((float4)(amd_unpack1(p2.s0), amd_unpack1(p3.s0), amd_unpack2(p1.s0), amd_unpack2(p2.s0)));\n"
			"  (*p0).s2 = amd_pack((float4)(amd_unpack2(p3.s0), amd_unpack3(p1.s0), amd_unpack3(p2.s0), amd_unpack3(p3.s0)));\n"
			"  (*p0).s3 = amd_pack((float4)(amd_unpack0(p1.s1), amd_unpack0(p2.s1), amd_unpack0(p3.s1), amd_unpack1(p1.s1)));\n"
			"  (*p0).s4 = amd_pack((float4)(amd_unpack1(p2.s1), amd_unpack1(p3.s1), amd_unpack2(p1.s1), amd_unpack2(p2.s1)));\n"
			"  (*p0).s5 = amd_pack((float4)(amd_unpack2(p3.s1), amd_unpack3(p1.s1), amd_unpack3(p2.s1), amd_unpack3(p3.s1)));\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out  = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		AgoData * inp3 = node->paramList[3];
		out->u.img.rect_valid.start_x = max(max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x), inp3->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y), inp3->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x), inp3->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y), inp3->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_ChannelCombine_U32_U8U8U8_UYVY(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg1 = node->paramList[1];
		AgoData * iImg2 = node->paramList[2];
		AgoData * iImg3 = node->paramList[3];
		if (HafCpu_ChannelCombine_U32_U8U8U8_UYVY(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, 
												  iImg1->buffer, iImg1->u.img.stride_in_bytes, iImg2->buffer, iImg2->u.img.stride_in_bytes,
												  iImg3->buffer, iImg3->u.img.stride_in_bytes)) 
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_U8 || node->paramList[2]->u.img.format != VX_DF_IMAGE_U8 || node->paramList[3]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != (node->paramList[2]->u.img.width << 1) || height != node->paramList[2]->u.img.height || 
			                          width != (node->paramList[3]->u.img.width << 1) || height != node->paramList[3]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_UYVY;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ChannelCombine_U32_422(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ChannelCombine_U32_U8U8U8_YUYV(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg1 = node->paramList[1];
		AgoData * iImg2 = node->paramList[2];
		AgoData * iImg3 = node->paramList[3];
		if (HafCpu_ChannelCombine_U32_U8U8U8_YUYV(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, 
			                                      iImg1->buffer, iImg1->u.img.stride_in_bytes, iImg2->buffer, iImg2->u.img.stride_in_bytes, 
												  iImg3->buffer, iImg3->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_U8 || node->paramList[2]->u.img.format != VX_DF_IMAGE_U8 || node->paramList[3]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != (node->paramList[2]->u.img.width << 1) || height != node->paramList[2]->u.img.height ||
			width != (node->paramList[3]->u.img.width << 1) || height != node->paramList[3]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_YUYV;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ChannelCombine_U32_422(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ChannelCombine_U32_U8U8U8U8_RGBX(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg1 = node->paramList[1];
		AgoData * iImg2 = node->paramList[2];
		AgoData * iImg3 = node->paramList[3];
		AgoData * iImg4 = node->paramList[4];
		if (HafCpu_ChannelCombine_U32_U8U8U8U8_RGBX(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, 
			                                        iImg1->buffer, iImg1->u.img.stride_in_bytes, iImg2->buffer, iImg2->u.img.stride_in_bytes, 
													iImg3->buffer, iImg3->u.img.stride_in_bytes, iImg4->buffer, iImg4->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_4IN(node, VX_DF_IMAGE_RGBX, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U32x8 * p0, U8x8 p1, U8x8 p2, U8x8 p3, U8x8 p4)\n"
			"{\n"
			"  U32x8 r;\n"
			"  r.s0 = amd_pack((float4)(amd_unpack0(p1.s0), amd_unpack0(p2.s0), amd_unpack0(p3.s0), amd_unpack0(p4.s0)));\n"
			"  r.s1 = amd_pack((float4)(amd_unpack1(p1.s0), amd_unpack1(p2.s0), amd_unpack1(p3.s0), amd_unpack1(p4.s0)));\n"
			"  r.s2 = amd_pack((float4)(amd_unpack2(p1.s0), amd_unpack2(p2.s0), amd_unpack2(p3.s0), amd_unpack2(p4.s0)));\n"
			"  r.s3 = amd_pack((float4)(amd_unpack3(p1.s0), amd_unpack3(p2.s0), amd_unpack3(p3.s0), amd_unpack3(p4.s0)));\n"
			"  r.s4 = amd_pack((float4)(amd_unpack0(p1.s1), amd_unpack0(p2.s1), amd_unpack0(p3.s1), amd_unpack0(p4.s1)));\n"
			"  r.s5 = amd_pack((float4)(amd_unpack1(p1.s1), amd_unpack1(p2.s1), amd_unpack1(p3.s1), amd_unpack1(p4.s1)));\n"
			"  r.s6 = amd_pack((float4)(amd_unpack2(p1.s1), amd_unpack2(p2.s1), amd_unpack2(p3.s1), amd_unpack2(p4.s1)));\n"
			"  r.s7 = amd_pack((float4)(amd_unpack3(p1.s1), amd_unpack3(p2.s1), amd_unpack3(p3.s1), amd_unpack3(p4.s1)));\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out  = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		AgoData * inp3 = node->paramList[3];
		AgoData * inp4 = node->paramList[4];
		out->u.img.rect_valid.start_x = max(max(max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x), inp3->u.img.rect_valid.start_x), inp4->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(max(max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y), inp3->u.img.rect_valid.start_y), inp4->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(min(min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x), inp3->u.img.rect_valid.end_x), inp4->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(min(min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y), inp3->u.img.rect_valid.end_y), inp4->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Mul_U24_U24U8_Sat_Round(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		// not implemented yet
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN_S(node, VX_DF_IMAGE_RGB, VX_DF_IMAGE_RGB, VX_DF_IMAGE_U8, VX_TYPE_FLOAT32);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s(U24x8 * p0, U24x8 p1, U8x8 p2, float p3)\n"
			"{\n"
			"  U24x8 r;\n"
			"  float4 f; float m3;\n"
			"  m3 = p3 * amd_unpack0(p2.s0);\n"
			"  f.s0 = m3 * amd_unpack0(p1.s0);\n"
			"  f.s1 = m3 * amd_unpack1(p1.s0);\n"
			"  f.s2 = m3 * amd_unpack2(p1.s0);\n"
			"  m3 = p3 * amd_unpack1(p2.s0);\n"
			"  f.s3 = m3 * amd_unpack3(p1.s0);\n"
			"  r.s0 = amd_pack(f);\n"
			"  f.s0 = m3 * amd_unpack0(p1.s1);\n"
			"  f.s1 = m3 * amd_unpack1(p1.s1);\n"
			"  m3 = p3 * amd_unpack2(p2.s0);\n"
			"  f.s2 = m3 * amd_unpack2(p1.s1);\n"
			"  f.s3 = m3 * amd_unpack3(p1.s1);\n"
			"  r.s1 = amd_pack(f);\n"
			"  f.s0 = m3 * amd_unpack0(p1.s2);\n"
			"  m3 = p3 * amd_unpack3(p2.s0);\n"
			"  f.s1 = m3 * amd_unpack1(p1.s2);\n"
			"  f.s2 = m3 * amd_unpack2(p1.s2);\n"
			"  f.s3 = m3 * amd_unpack3(p1.s2);\n"
			"  r.s2 = amd_pack(f);\n"
			"  m3 = p3 * amd_unpack0(p2.s1);\n"
			"  f.s0 = m3 * amd_unpack0(p1.s3);\n"
			"  f.s1 = m3 * amd_unpack1(p1.s3);\n"
			"  f.s2 = m3 * amd_unpack2(p1.s3);\n"
			"  m3 = p3 * amd_unpack1(p2.s1);\n"
			"  f.s3 = m3 * amd_unpack3(p1.s3);\n"
			"  r.s3 = amd_pack(f);\n"
			"  f.s0 = m3 * amd_unpack0(p1.s4);\n"
			"  f.s1 = m3 * amd_unpack1(p1.s4);\n"
			"  m3 = p3 * amd_unpack2(p2.s1);\n"
			"  f.s2 = m3 * amd_unpack2(p1.s4);\n"
			"  f.s3 = m3 * amd_unpack3(p1.s4);\n"
			"  r.s4 = amd_pack(f);\n"
			"  f.s0 = m3 * amd_unpack0(p1.s5);\n"
			"  m3 = p3 * amd_unpack3(p2.s1);\n"
			"  f.s1 = m3 * amd_unpack1(p1.s5);\n"
			"  f.s2 = m3 * amd_unpack2(p1.s5);\n"
			"  f.s3 = m3 * amd_unpack3(p1.s5);\n"
			"  r.s5 = amd_pack(f);\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
#if ENABLE_OPENCL
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif			
			;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_Mul_U32_U32U8_Sat_Round(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		// not implemented yet
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN_S(node, VX_DF_IMAGE_RGBX, VX_DF_IMAGE_RGBX, VX_DF_IMAGE_U8, VX_TYPE_FLOAT32);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s(U32x8 * p0, U32x8 p1, U8x8 p2, float p3)\n"
			"{\n"
			"  U32x8 r;\n"
			"  float4 f; float m3;\n"
			"  m3 = p3 * amd_unpack0(p2.s0);\n"
			"  f.s0 = m3 * amd_unpack0(p1.s0);\n"
			"  f.s1 = m3 * amd_unpack1(p1.s0);\n"
			"  f.s2 = m3 * amd_unpack2(p1.s0);\n"
			"  f.s3 = m3 * amd_unpack3(p1.s0);\n"
			"  r.s0 = amd_pack(f);\n"
			"  m3 = p3 * amd_unpack1(p2.s0);\n"
			"  f.s0 = m3 * amd_unpack0(p1.s1);\n"
			"  f.s1 = m3 * amd_unpack1(p1.s1);\n"
			"  f.s2 = m3 * amd_unpack2(p1.s1);\n"
			"  f.s3 = m3 * amd_unpack3(p1.s1);\n"
			"  r.s1 = amd_pack(f);\n"
			"  m3 = p3 * amd_unpack2(p2.s0);\n"
			"  f.s0 = m3 * amd_unpack0(p1.s2);\n"
			"  f.s1 = m3 * amd_unpack1(p1.s2);\n"
			"  f.s2 = m3 * amd_unpack2(p1.s2);\n"
			"  f.s3 = m3 * amd_unpack3(p1.s2);\n"
			"  r.s2 = amd_pack(f);\n"
			"  m3 = p3 * amd_unpack3(p2.s0);\n"
			"  f.s0 = m3 * amd_unpack0(p1.s3);\n"
			"  f.s1 = m3 * amd_unpack1(p1.s3);\n"
			"  f.s2 = m3 * amd_unpack2(p1.s3);\n"
			"  f.s3 = m3 * amd_unpack3(p1.s3);\n"
			"  r.s3 = amd_pack(f);\n"
			"  m3 = p3 * amd_unpack0(p2.s1);\n"
			"  f.s0 = m3 * amd_unpack0(p1.s4);\n"
			"  f.s1 = m3 * amd_unpack1(p1.s4);\n"
			"  f.s2 = m3 * amd_unpack2(p1.s4);\n"
			"  f.s3 = m3 * amd_unpack3(p1.s4);\n"
			"  r.s4 = amd_pack(f);\n"
			"  m3 = p3 * amd_unpack1(p2.s1);\n"
			"  f.s0 = m3 * amd_unpack0(p1.s5);\n"
			"  f.s1 = m3 * amd_unpack1(p1.s5);\n"
			"  f.s2 = m3 * amd_unpack2(p1.s5);\n"
			"  f.s3 = m3 * amd_unpack3(p1.s5);\n"
			"  r.s5 = amd_pack(f);\n"
			"  m3 = p3 * amd_unpack2(p2.s1);\n"
			"  f.s0 = m3 * amd_unpack0(p1.s6);\n"
			"  f.s1 = m3 * amd_unpack1(p1.s6);\n"
			"  f.s2 = m3 * amd_unpack2(p1.s6);\n"
			"  f.s3 = m3 * amd_unpack3(p1.s6);\n"
			"  r.s6 = amd_pack(f);\n"
			"  m3 = p3 * amd_unpack3(p2.s1);\n"
			"  f.s0 = m3 * amd_unpack0(p1.s7);\n"
			"  f.s1 = m3 * amd_unpack1(p1.s7);\n"
			"  f.s2 = m3 * amd_unpack2(p1.s7);\n"
			"  f.s3 = m3 * amd_unpack3(p1.s7);\n"
			"  r.s7 = amd_pack(f);\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
#if ENABLE_OPENCL		
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif			
			;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp1 = node->paramList[1];
		AgoData * inp2 = node->paramList[2];
		out->u.img.rect_valid.start_x = max(inp1->u.img.rect_valid.start_x, inp2->u.img.rect_valid.start_x);
		out->u.img.rect_valid.start_y = max(inp1->u.img.rect_valid.start_y, inp2->u.img.rect_valid.start_y);
		out->u.img.rect_valid.end_x = min(inp1->u.img.rect_valid.end_x, inp2->u.img.rect_valid.end_x);
		out->u.img.rect_valid.end_y = min(inp1->u.img.rect_valid.end_y, inp2->u.img.rect_valid.end_y);
	}
	return status;
}

int agoKernel_ColorConvert_RGB_RGBX(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ColorConvert_RGB_RGBX(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
										 iImg->buffer, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_RGB, VX_DF_IMAGE_RGBX);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U24x8 * p0, U32x8 p1)\n"
			"{\n"
			"  (*p0).s0 = amd_pack((float4)(amd_unpack0(p1.s0), amd_unpack1(p1.s0), amd_unpack2(p1.s0), amd_unpack0(p1.s1)));\n"
			"  (*p0).s1 = amd_pack((float4)(amd_unpack1(p1.s1), amd_unpack2(p1.s1), amd_unpack0(p1.s2), amd_unpack1(p1.s2)));\n"
			"  (*p0).s2 = amd_pack((float4)(amd_unpack2(p1.s2), amd_unpack0(p1.s3), amd_unpack1(p1.s3), amd_unpack2(p1.s3)));\n"
			"  (*p0).s3 = amd_pack((float4)(amd_unpack0(p1.s4), amd_unpack1(p1.s4), amd_unpack2(p1.s4), amd_unpack0(p1.s5)));\n"
			"  (*p0).s4 = amd_pack((float4)(amd_unpack1(p1.s5), amd_unpack2(p1.s5), amd_unpack0(p1.s6), amd_unpack1(p1.s6)));\n"
			"  (*p0).s5 = amd_pack((float4)(amd_unpack2(p1.s6), amd_unpack0(p1.s7), amd_unpack1(p1.s7), amd_unpack2(p1.s7)));\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorConvert_RGB_UYVY(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ColorConvert_RGB_UYVY(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
										 iImg->buffer, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_RGB, VX_DF_IMAGE_UYVY);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ColorConvert(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
					| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif				
					;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorConvert_RGB_YUYV(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ColorConvert_RGB_YUYV(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
										 iImg->buffer, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_RGB, VX_DF_IMAGE_YUYV);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ColorConvert(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
			| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL			
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif			
			;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorConvert_RGB_IYUV(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg1 = node->paramList[1];
		AgoData * iImg2 = node->paramList[2];
		AgoData * iImg3 = node->paramList[3];
		if (HafCpu_ColorConvert_RGB_IYUV(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
										 iImg1->buffer, iImg1->u.img.stride_in_bytes, iImg2->buffer, iImg2->u.img.stride_in_bytes,
										 iImg3->buffer, iImg3->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_U8 || node->paramList[2]->u.img.format != VX_DF_IMAGE_U8 || node->paramList[3]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != (node->paramList[2]->u.img.width << 1) || (height != node->paramList[2]->u.img.height << 1) ||
									  width != (node->paramList[3]->u.img.width << 1) || (height != node->paramList[3]->u.img.height << 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_RGB;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ColorConvert(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
			| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL			
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif			
			;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorConvert_RGB_NV12(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg1 = node->paramList[1];
		AgoData * iImg2 = node->paramList[2];
		if (HafCpu_ColorConvert_RGB_NV12(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
										 iImg1->buffer, iImg1->u.img.stride_in_bytes, iImg2->buffer, iImg2->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_U8 || node->paramList[2]->u.img.format != VX_DF_IMAGE_U16)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != (node->paramList[2]->u.img.width << 1) || (height != node->paramList[2]->u.img.height << 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_RGB;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ColorConvert(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
			| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL			
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif			
			;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorConvert_RGB_NV21(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg1 = node->paramList[1];
		AgoData * iImg2 = node->paramList[2];
		if (HafCpu_ColorConvert_RGB_NV21(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
										 iImg1->buffer, iImg1->u.img.stride_in_bytes, iImg2->buffer, iImg2->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_U8 || node->paramList[2]->u.img.format != VX_DF_IMAGE_U16)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != (node->paramList[2]->u.img.width << 1) || (height != node->paramList[2]->u.img.height << 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_RGB;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ColorConvert(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
			| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif
			;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorConvert_RGBX_RGB(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ColorConvert_RGBX_RGB(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
										 iImg->buffer, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_RGBX, VX_DF_IMAGE_RGB);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U32x8 * p0, U24x8 p1)\n"
			"{\n"
			"  U32x8 r;\n"
			"  r.s0 = amd_pack((float4)(amd_unpack0(p1.s0), amd_unpack1(p1.s0), amd_unpack2(p1.s0), 255.0f));\n"
			"  r.s1 = amd_pack((float4)(amd_unpack3(p1.s0), amd_unpack0(p1.s1), amd_unpack1(p1.s1), 255.0f));\n"
			"  r.s2 = amd_pack((float4)(amd_unpack2(p1.s1), amd_unpack3(p1.s1), amd_unpack0(p1.s2), 255.0f));\n"
			"  r.s3 = amd_pack((float4)(amd_unpack1(p1.s2), amd_unpack2(p1.s2), amd_unpack3(p1.s2), 255.0f));\n"
			"  r.s4 = amd_pack((float4)(amd_unpack0(p1.s3), amd_unpack1(p1.s3), amd_unpack2(p1.s3), 255.0f));\n"
			"  r.s5 = amd_pack((float4)(amd_unpack3(p1.s3), amd_unpack0(p1.s4), amd_unpack1(p1.s4), 255.0f));\n"
			"  r.s6 = amd_pack((float4)(amd_unpack2(p1.s4), amd_unpack3(p1.s4), amd_unpack0(p1.s5), 255.0f));\n"
			"  r.s7 = amd_pack((float4)(amd_unpack1(p1.s5), amd_unpack2(p1.s5), amd_unpack3(p1.s5), 255.0f));\n"
			"  *p0 = r;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorConvert_RGBX_UYVY(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ColorConvert_RGBX_UYVY(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
										  iImg->buffer, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_RGBX, VX_DF_IMAGE_UYVY);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ColorConvert(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
			| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL			
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif			
			;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorConvert_RGBX_YUYV(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ColorConvert_RGBX_YUYV(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
										  iImg->buffer, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_RGBX, VX_DF_IMAGE_YUYV);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ColorConvert(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
			| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL			
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif			
			;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorConvert_RGBX_IYUV(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg1 = node->paramList[1];
		AgoData * iImg2 = node->paramList[2];
		AgoData * iImg3 = node->paramList[3];
		if (HafCpu_ColorConvert_RGBX_IYUV(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
										  iImg1->buffer, iImg1->u.img.stride_in_bytes, iImg2->buffer, iImg2->u.img.stride_in_bytes,
										  iImg3->buffer, iImg3->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_U8 || node->paramList[2]->u.img.format != VX_DF_IMAGE_U8 || node->paramList[3]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != (node->paramList[2]->u.img.width << 1) || (height != node->paramList[2]->u.img.height << 1) ||
									  width != (node->paramList[3]->u.img.width << 1) || (height != node->paramList[3]->u.img.height << 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_RGBX;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ColorConvert(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
			| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL			
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif			
			;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorConvert_RGBX_NV12(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg1 = node->paramList[1];
		AgoData * iImg2 = node->paramList[2];
		if (HafCpu_ColorConvert_RGBX_NV12(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
										  iImg1->buffer, iImg1->u.img.stride_in_bytes, iImg2->buffer, iImg2->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_U8 || node->paramList[2]->u.img.format != VX_DF_IMAGE_U16)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != (node->paramList[2]->u.img.width << 1) || (height != node->paramList[2]->u.img.height << 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_RGBX;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ColorConvert(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
			| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL			
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif			
			;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorConvert_RGBX_NV21(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg1 = node->paramList[1];
		AgoData * iImg2 = node->paramList[2];
		if (HafCpu_ColorConvert_RGBX_NV21(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
										  iImg1->buffer, iImg1->u.img.stride_in_bytes, iImg2->buffer, iImg2->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_U8 || node->paramList[2]->u.img.format != VX_DF_IMAGE_U16)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || width != (node->paramList[2]->u.img.width << 1) || (height != node->paramList[2]->u.img.height << 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_RGBX;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ColorConvert(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
			| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL			
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif			
			;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorConvert_YUV4_RGB(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImgY = node->paramList[0];
		AgoData * oImgU = node->paramList[1];
		AgoData * oImgV = node->paramList[2];
		AgoData * iImg = node->paramList[3];
		if (HafCpu_ColorConvert_YUV4_RGB(oImgY->u.img.width, oImgY->u.img.height, oImgY->buffer, oImgY->u.img.stride_in_bytes,
										 oImgU->buffer, oImgU->u.img.stride_in_bytes, oImgV->buffer, oImgV->u.img.stride_in_bytes,
										 iImg->buffer, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[3]->u.img.width;
		vx_uint32 height = node->paramList[3]->u.img.height;
		if (node->paramList[3]->u.img.format != VX_DF_IMAGE_RGB)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		meta = &node->metaList[1];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		meta = &node->metaList[2];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_ColorConvert_Y_RGB(node->opencl_code);
		agoCodeGenOpenCL_ColorConvert_U_RGB(node->opencl_code);
		agoCodeGenOpenCL_ColorConvert_V_RGB(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 * p1, U8x8 * p2, U24x8 p3)\n"
			"{\n"
			"  ColorConvert_Y_RGB(p0, p3);\n"
			"  ColorConvert_U_RGB(p1, p3);\n"
			"  ColorConvert_V_RGB(p2, p3);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out1 = node->paramList[0];
		AgoData * out2 = node->paramList[1];
		AgoData * out3 = node->paramList[2];
		AgoData * inp = node->paramList[3];
		out1->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out1->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out1->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out1->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
		out2->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out2->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out2->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out2->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
		out3->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out3->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out3->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out3->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorConvert_YUV4_RGBX(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImgY = node->paramList[0];
		AgoData * oImgU = node->paramList[1];
		AgoData * oImgV = node->paramList[2];
		AgoData * iImg = node->paramList[3];
		if (HafCpu_ColorConvert_YUV4_RGBX(oImgY->u.img.width, oImgY->u.img.height, oImgY->buffer, oImgY->u.img.stride_in_bytes,
										 oImgU->buffer, oImgU->u.img.stride_in_bytes, oImgV->buffer, oImgV->u.img.stride_in_bytes,
										 iImg->buffer, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[3]->u.img.width;
		vx_uint32 height = node->paramList[3]->u.img.height;
		if (node->paramList[3]->u.img.format != VX_DF_IMAGE_RGBX)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		meta = &node->metaList[1];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		meta = &node->metaList[2];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_ColorConvert_Y_RGBX(node->opencl_code);
		agoCodeGenOpenCL_ColorConvert_U_RGBX(node->opencl_code);
		agoCodeGenOpenCL_ColorConvert_V_RGBX(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s (U8x8 * p0, U8x8 * p1, U8x8 * p2, U32x8 p3)\n"
			"{\n"
			"  ColorConvert_Y_RGBX(p0, p3);\n"
			"  ColorConvert_U_RGBX(p1, p3);\n"
			"  ColorConvert_V_RGBX(p2, p3);\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out1 = node->paramList[0];
		AgoData * out2 = node->paramList[1];
		AgoData * out3 = node->paramList[2];
		AgoData * inp = node->paramList[3];
		out1->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out1->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out1->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out1->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
		out2->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out2->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out2->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out2->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
		out3->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out3->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out3->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out3->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ScaleUp2x2_U8_U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ScaleUp2x2_U8_U8(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
									iImg->buffer, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes and format
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width << 1;
		meta->data.u.img.height = height << 1;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_FormatConvert_Chroma(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x << 1;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y << 1;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x << 1;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y << 1;
	}
	return status;
}

int agoKernel_FormatConvert_UV_UV12(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImgU = node->paramList[0];
		AgoData * oImgV = node->paramList[1];
		AgoData * iImgC = node->paramList[2];
		if (HafCpu_FormatConvert_UV_UV12(oImgU->u.img.width, oImgU->u.img.height, oImgU->buffer, oImgU->u.img.stride_in_bytes,
										 oImgV->buffer, oImgV->u.img.stride_in_bytes, iImgC->buffer, iImgC->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[2]->u.img.width;
		vx_uint32 height = node->paramList[2]->u.img.height;
		if (node->paramList[2]->u.img.format != VX_DF_IMAGE_U16)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes and format
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width << 1;
		meta->data.u.img.height = height << 1;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		meta = &node->metaList[1];
		meta->data.u.img.width = width << 1;
		meta->data.u.img.height = height << 1;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_FormatConvert_Chroma(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out1 = node->paramList[0];
		AgoData * out2 = node->paramList[1];
		AgoData * inp = node->paramList[2];
		out1->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x << 1;
		out1->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y << 1;
		out1->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x << 1;
		out1->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y << 1;
		out2->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x << 1;
		out2->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y << 1;
		out2->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x << 1;
		out2->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y << 1;
	}
	return status;
}

int agoKernel_ColorConvert_IYUV_RGB(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImgY = node->paramList[0];
		AgoData * oImgU = node->paramList[1];
		AgoData * oImgV = node->paramList[2];
		AgoData * iImg = node->paramList[3];
		if (HafCpu_ColorConvert_IYUV_RGB(oImgY->u.img.width, oImgY->u.img.height, oImgY->buffer, oImgY->u.img.stride_in_bytes,
										 oImgU->buffer, oImgU->u.img.stride_in_bytes, oImgV->buffer, oImgV->u.img.stride_in_bytes,
										 iImg->buffer, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[3]->u.img.width;
		vx_uint32 height = node->paramList[3]->u.img.height;
		if (node->paramList[3]->u.img.format != VX_DF_IMAGE_RGB)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || (width & 1) || (height & 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		meta = &node->metaList[1];
		meta->data.u.img.width = width >> 1;
		meta->data.u.img.height = height >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		meta = &node->metaList[2];
		meta->data.u.img.width = width >> 1;
		meta->data.u.img.height = height >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ColorConvert(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
			| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL			
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif			
			;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out1 = node->paramList[0];
		AgoData * out2 = node->paramList[1];
		AgoData * out3 = node->paramList[2];
		AgoData * inp = node->paramList[3];
		out1->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out1->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out1->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out1->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
		out2->u.img.rect_valid.start_x = (inp->u.img.rect_valid.start_x + 1) >> 1;
		out2->u.img.rect_valid.start_y = (inp->u.img.rect_valid.start_y + 1) >> 1;
		out2->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x >> 1;
		out2->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y >> 1;
		out3->u.img.rect_valid.start_x = (inp->u.img.rect_valid.start_x + 1) >> 1;
		out3->u.img.rect_valid.start_y = (inp->u.img.rect_valid.start_y + 1) >> 1;
		out3->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x >> 1;
		out3->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y >> 1;
	}
	return status;
}

int agoKernel_ColorConvert_IYUV_RGBX(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImgY = node->paramList[0];
		AgoData * oImgU = node->paramList[1];
		AgoData * oImgV = node->paramList[2];
		AgoData * iImg = node->paramList[3];
		if (HafCpu_ColorConvert_IYUV_RGBX(oImgY->u.img.width, oImgY->u.img.height, oImgY->buffer, oImgY->u.img.stride_in_bytes,
										  oImgU->buffer, oImgU->u.img.stride_in_bytes, oImgV->buffer, oImgV->u.img.stride_in_bytes,
										  iImg->buffer, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[3]->u.img.width;
		vx_uint32 height = node->paramList[3]->u.img.height;
		if (node->paramList[3]->u.img.format != VX_DF_IMAGE_RGBX)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || (width & 1) || (height & 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		meta = &node->metaList[1];
		meta->data.u.img.width = width >> 1;
		meta->data.u.img.height = height >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		meta = &node->metaList[2];
		meta->data.u.img.width = width >> 1;
		meta->data.u.img.height = height >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ColorConvert(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
			| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL			
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif			
			;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out1 = node->paramList[0];
		AgoData * out2 = node->paramList[1];
		AgoData * out3 = node->paramList[2];
		AgoData * inp = node->paramList[3];
		out1->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out1->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out1->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out1->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
		out2->u.img.rect_valid.start_x = (inp->u.img.rect_valid.start_x + 1) >> 1;
		out2->u.img.rect_valid.start_y = (inp->u.img.rect_valid.start_y + 1) >> 1;
		out2->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x >> 1;
		out2->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y >> 1;
		out3->u.img.rect_valid.start_x = (inp->u.img.rect_valid.start_x + 1) >> 1;
		out3->u.img.rect_valid.start_y = (inp->u.img.rect_valid.start_y + 1) >> 1;
		out3->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x >> 1;
		out3->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y >> 1;
	}
	return status;
}

int agoKernel_FormatConvert_IYUV_UYVY(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImgY = node->paramList[0];
		AgoData * oImgU = node->paramList[1];
		AgoData * oImgV = node->paramList[2];
		AgoData * iImg = node->paramList[3];
		if (HafCpu_FormatConvert_IYUV_UYVY(oImgY->u.img.width, oImgY->u.img.height, oImgY->buffer, oImgY->u.img.stride_in_bytes,
										  oImgU->buffer, oImgU->u.img.stride_in_bytes, oImgV->buffer, oImgV->u.img.stride_in_bytes,
										  iImg->buffer, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[3]->u.img.width;
		vx_uint32 height = node->paramList[3]->u.img.height;
		if (node->paramList[3]->u.img.format != VX_DF_IMAGE_UYVY)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || (width & 1) || (height & 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		meta = &node->metaList[1];
		meta->data.u.img.width = width >> 1;
		meta->data.u.img.height = height >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		meta = &node->metaList[2];
		meta->data.u.img.width = width >> 1;
		meta->data.u.img.height = height >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_FormatConvert_420_422(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out1 = node->paramList[0];
		AgoData * out2 = node->paramList[1];
		AgoData * out3 = node->paramList[2];
		AgoData * inp = node->paramList[3];
		out1->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out1->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out1->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out1->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
		out2->u.img.rect_valid.start_x = (inp->u.img.rect_valid.start_x + 1) >> 1;
		out2->u.img.rect_valid.start_y = (inp->u.img.rect_valid.start_y + 1) >> 1;
		out2->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x >> 1;
		out2->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y >> 1;
		out3->u.img.rect_valid.start_x = (inp->u.img.rect_valid.start_x + 1) >> 1;
		out3->u.img.rect_valid.start_y = (inp->u.img.rect_valid.start_y + 1) >> 1;
		out3->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x >> 1;
		out3->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y >> 1;
	}
	return status;
}

int agoKernel_FormatConvert_IYUV_YUYV(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImgY = node->paramList[0];
		AgoData * oImgU = node->paramList[1];
		AgoData * oImgV = node->paramList[2];
		AgoData * iImg = node->paramList[3];
		if (HafCpu_FormatConvert_IYUV_YUYV(oImgY->u.img.width, oImgY->u.img.height, oImgY->buffer, oImgY->u.img.stride_in_bytes,
										  oImgU->buffer, oImgU->u.img.stride_in_bytes, oImgV->buffer, oImgV->u.img.stride_in_bytes,
										  iImg->buffer, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[3]->u.img.width;
		vx_uint32 height = node->paramList[3]->u.img.height;
		if (node->paramList[3]->u.img.format != VX_DF_IMAGE_YUYV)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || (width & 1) || (height & 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes are same as input image size
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		meta = &node->metaList[1];
		meta->data.u.img.width = width >> 1;
		meta->data.u.img.height = height >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		meta = &node->metaList[2];
		meta->data.u.img.width = width >> 1;
		meta->data.u.img.height = height >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_FormatConvert_420_422(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out1 = node->paramList[0];
		AgoData * out2 = node->paramList[1];
		AgoData * out3 = node->paramList[2];
		AgoData * inp = node->paramList[3];
		out1->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out1->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out1->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out1->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
		out2->u.img.rect_valid.start_x = (inp->u.img.rect_valid.start_x + 1) >> 1;
		out2->u.img.rect_valid.start_y = (inp->u.img.rect_valid.start_y + 1) >> 1;
		out2->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x >> 1;
		out2->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y >> 1;
		out3->u.img.rect_valid.start_x = (inp->u.img.rect_valid.start_x + 1) >> 1;
		out3->u.img.rect_valid.start_y = (inp->u.img.rect_valid.start_y + 1) >> 1;
		out3->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x >> 1;
		out3->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y >> 1;
	}
	return status;
}

int agoKernel_FormatConvert_IUV_UV12(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImgU = node->paramList[0];
		AgoData * oImgV = node->paramList[1];
		AgoData * iImgC = node->paramList[2];
		if (HafCpu_FormatConvert_IUV_UV12(oImgU->u.img.width, oImgU->u.img.height, oImgU->buffer, oImgU->u.img.stride_in_bytes,
										  oImgV->buffer, oImgV->u.img.stride_in_bytes, iImgC->buffer, iImgC->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[2]->u.img.width;
		vx_uint32 height = node->paramList[2]->u.img.height;
		if (node->paramList[2]->u.img.format != VX_DF_IMAGE_U16)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes and format
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		meta = &node->metaList[1];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_FormatConvert_Chroma(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out1 = node->paramList[0];
		AgoData * out2 = node->paramList[1];
		AgoData * inp = node->paramList[2];
		out1->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out1->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out1->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out1->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
		out2->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out2->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out2->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out2->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorConvert_NV12_RGB(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImgY = node->paramList[0];
		AgoData * oImgC = node->paramList[1];
		AgoData * iImg  = node->paramList[2];
		if (HafCpu_ColorConvert_NV12_RGB(oImgY->u.img.width, oImgY->u.img.height, oImgY->buffer, oImgY->u.img.stride_in_bytes,
										 oImgC->buffer, oImgC->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[2]->u.img.width;
		vx_uint32 height = node->paramList[2]->u.img.height;
		if (node->paramList[2]->u.img.format != VX_DF_IMAGE_RGB)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || (width & 1) || (height & 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes and format
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		meta = &node->metaList[1];
		meta->data.u.img.width = width >> 1;
		meta->data.u.img.height = height >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U16;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ColorConvert(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
			| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL			
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif			
			;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out1 = node->paramList[0];
		AgoData * out2 = node->paramList[1];
		AgoData * inp = node->paramList[2];
		out1->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out1->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out1->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out1->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
		out2->u.img.rect_valid.start_x = (inp->u.img.rect_valid.start_x + 1) >> 1;
		out2->u.img.rect_valid.start_y = (inp->u.img.rect_valid.start_y + 1) >> 1;
		out2->u.img.rect_valid.end_x = (inp->u.img.rect_valid.end_x + 1) >> 1;
		out2->u.img.rect_valid.end_y = (inp->u.img.rect_valid.end_y + 1) >> 1;
	}
	return status;
}

int agoKernel_ColorConvert_NV12_RGBX(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImgY = node->paramList[0];
		AgoData * oImgC = node->paramList[1];
		AgoData * iImg  = node->paramList[2];
		if (HafCpu_ColorConvert_NV12_RGBX(oImgY->u.img.width, oImgY->u.img.height, oImgY->buffer, oImgY->u.img.stride_in_bytes,
										  oImgC->buffer, oImgC->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[2]->u.img.width;
		vx_uint32 height = node->paramList[2]->u.img.height;
		if (node->paramList[2]->u.img.format != VX_DF_IMAGE_RGBX)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || (width & 1) || (height & 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes and format
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		meta = &node->metaList[1];
		meta->data.u.img.width = width >> 1;
		meta->data.u.img.height = height >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U16;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ColorConvert(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
			| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL			
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif			
			;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out1 = node->paramList[0];
		AgoData * out2 = node->paramList[1];
		AgoData * inp = node->paramList[2];
		out1->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out1->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out1->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out1->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
		out2->u.img.rect_valid.start_x = (inp->u.img.rect_valid.start_x + 1) >> 1;
		out2->u.img.rect_valid.start_y = (inp->u.img.rect_valid.start_y + 1) >> 1;
		out2->u.img.rect_valid.end_x = (inp->u.img.rect_valid.end_x + 1) >> 1;
		out2->u.img.rect_valid.end_y = (inp->u.img.rect_valid.end_y + 1) >> 1;
	}
	return status;
}

int agoKernel_FormatConvert_NV12_UYVY(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImgY = node->paramList[0];
		AgoData * oImgC = node->paramList[1];
		AgoData * iImg  = node->paramList[2];
		if (HafCpu_FormatConvert_NV12_UYVY(oImgY->u.img.width, oImgY->u.img.height, oImgY->buffer, oImgY->u.img.stride_in_bytes,
										   oImgC->buffer, oImgC->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[2]->u.img.width;
		vx_uint32 height = node->paramList[2]->u.img.height;
		if (node->paramList[2]->u.img.format != VX_DF_IMAGE_UYVY)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || (width & 1) || (height & 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes and format
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		meta = &node->metaList[1];
		meta->data.u.img.width = width >> 1;
		meta->data.u.img.height = height >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U16;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_FormatConvert_420_422(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out1 = node->paramList[0];
		AgoData * out2 = node->paramList[1];
		AgoData * inp = node->paramList[2];
		out1->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out1->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out1->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out1->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
		out2->u.img.rect_valid.start_x = (inp->u.img.rect_valid.start_x + 1) >> 1;
		out2->u.img.rect_valid.start_y = (inp->u.img.rect_valid.start_y + 1) >> 1;
		out2->u.img.rect_valid.end_x = (inp->u.img.rect_valid.end_x + 1) >> 1;
		out2->u.img.rect_valid.end_y = (inp->u.img.rect_valid.end_y + 1) >> 1;
	}
	return status;
}

int agoKernel_FormatConvert_NV12_YUYV(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImgY = node->paramList[0];
		AgoData * oImgC = node->paramList[1];
		AgoData * iImg  = node->paramList[2];
		if (HafCpu_FormatConvert_NV12_YUYV(oImgY->u.img.width, oImgY->u.img.height, oImgY->buffer, oImgY->u.img.stride_in_bytes,
										   oImgC->buffer, oImgC->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[2]->u.img.width;
		vx_uint32 height = node->paramList[2]->u.img.height;
		if (node->paramList[2]->u.img.format != VX_DF_IMAGE_YUYV)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || (width & 1) || (height & 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes and format
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		meta = &node->metaList[1];
		meta->data.u.img.width = width >> 1;
		meta->data.u.img.height = height >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U16;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_FormatConvert_420_422(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out1 = node->paramList[0];
		AgoData * out2 = node->paramList[1];
		AgoData * inp = node->paramList[2];
		out1->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out1->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out1->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out1->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
		out2->u.img.rect_valid.start_x = (inp->u.img.rect_valid.start_x + 1) >> 1;
		out2->u.img.rect_valid.start_y = (inp->u.img.rect_valid.start_y + 1) >> 1;
		out2->u.img.rect_valid.end_x = (inp->u.img.rect_valid.end_x + 1) >> 1;
		out2->u.img.rect_valid.end_y = (inp->u.img.rect_valid.end_y + 1) >> 1;
	}
	return status;
}

int agoKernel_FormatConvert_UV12_IUV(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImgC = node->paramList[0];
		AgoData * iImgU = node->paramList[1];
		AgoData * iImgV = node->paramList[2];
		if (HafCpu_FormatConvert_UV12_IUV(oImgC->u.img.width, oImgC->u.img.height, oImgC->buffer, oImgC->u.img.stride_in_bytes,
										  iImgU->buffer, iImgU->u.img.stride_in_bytes, iImgV->buffer, iImgV->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_2IN(node, VX_DF_IMAGE_U16, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_FormatConvert_Chroma(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorConvert_Y_RGB(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ColorConvert_Y_RGB(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_RGB);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_ColorConvert_Y_RGB(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"#define %s ColorConvert_Y_RGB\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorConvert_Y_RGBX(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ColorConvert_Y_RGBX(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_RGBX);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_ColorConvert_Y_RGBX(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"#define %s ColorConvert_Y_RGBX\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorConvert_U_RGB(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ColorConvert_U_RGB(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_RGB);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_ColorConvert_U_RGB(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"#define %s ColorConvert_U_RGB\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorConvert_U_RGBX(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ColorConvert_U_RGBX(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_RGBX);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_ColorConvert_U_RGBX(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"#define %s ColorConvert_U_RGBX\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorConvert_V_RGB(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ColorConvert_V_RGB(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_RGB);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_ColorConvert_V_RGB(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"#define %s ColorConvert_V_RGB\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorConvert_V_RGBX(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ColorConvert_V_RGBX(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_RGBX);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		node->opencl_type = NODE_OPENCL_TYPE_REG2REG;
		agoCodeGenOpenCL_ColorConvert_V_RGBX(node->opencl_code);
		char textBuffer[2048];
		sprintf(textBuffer, OPENCL_FORMAT(
			"#define %s ColorConvert_V_RGBX\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_R2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorConvert_IU_RGB(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ColorConvert_IU_RGB(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_RGB)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || (width & 1) || (height & 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes and format
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width >> 1;
		meta->data.u.img.height = height >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ColorConvert(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
#if ENABLE_OPENCL		
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif			
			;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = (inp->u.img.rect_valid.start_x + 1) >> 1;
		out->u.img.rect_valid.start_y = (inp->u.img.rect_valid.start_y + 1) >> 1;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x >> 1;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y >> 1;
	}
	return status;
}

int agoKernel_ColorConvert_IU_RGBX(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ColorConvert_IU_RGBX(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_RGBX)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || (width & 1) || (height & 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes and format
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width >> 1;
		meta->data.u.img.height = height >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ColorConvert(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
#if ENABLE_OPENCL		
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif			
			;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = (inp->u.img.rect_valid.start_x + 1) >> 1;
		out->u.img.rect_valid.start_y = (inp->u.img.rect_valid.start_y + 1) >> 1;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x >> 1;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y >> 1;
	}
	return status;
}

int agoKernel_ColorConvert_IV_RGB(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ColorConvert_IV_RGB(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_RGB)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || (width & 1) || (height & 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes and format
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width >> 1;
		meta->data.u.img.height = height >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ColorConvert(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
#if ENABLE_OPENCL		
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif			
			;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = (inp->u.img.rect_valid.start_x + 1) >> 1;
		out->u.img.rect_valid.start_y = (inp->u.img.rect_valid.start_y + 1) >> 1;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x >> 1;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y >> 1;
	}
	return status;
}

int agoKernel_ColorConvert_IV_RGBX(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ColorConvert_IV_RGBX(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_RGBX)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || (width & 1) || (height & 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes and format
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width >> 1;
		meta->data.u.img.height = height >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ColorConvert(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
#if ENABLE_OPENCL		
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif			
			;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = (inp->u.img.rect_valid.start_x + 1) >> 1;
		out->u.img.rect_valid.start_y = (inp->u.img.rect_valid.start_y + 1) >> 1;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x >> 1;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y >> 1;
	}
	return status;
}

int agoKernel_ColorConvert_IUV_RGB(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImgU = node->paramList[0];
		AgoData * oImgV = node->paramList[1];
		AgoData * iImg = node->paramList[2];
		if (HafCpu_ColorConvert_IUV_RGB(oImgU->u.img.width, oImgU->u.img.height, oImgU->buffer, oImgU->u.img.stride_in_bytes, 
										oImgV->buffer, oImgV->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[2]->u.img.width;
		vx_uint32 height = node->paramList[2]->u.img.height;
		if (node->paramList[2]->u.img.format != VX_DF_IMAGE_RGB)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || (width & 1) || (height & 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes and format
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width >> 1;
		meta->data.u.img.height = height >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		meta = &node->metaList[1];
		meta->data.u.img.width = width >> 1;
		meta->data.u.img.height = height >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ColorConvert(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
#if ENABLE_OPENCL		
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif			
			;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out1 = node->paramList[0];
		AgoData * out2 = node->paramList[1];
		AgoData * inp = node->paramList[2];
		out1->u.img.rect_valid.start_x = (inp->u.img.rect_valid.start_x + 1) >> 1;
		out1->u.img.rect_valid.start_y = (inp->u.img.rect_valid.start_y + 1) >> 1;
		out1->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x >> 1;
		out1->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y >> 1;
		out2->u.img.rect_valid.start_x = (inp->u.img.rect_valid.start_x + 1) >> 1;
		out2->u.img.rect_valid.start_y = (inp->u.img.rect_valid.start_y + 1) >> 1;
		out2->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x >> 1;
		out2->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y >> 1;
	}
	return status;
}

int agoKernel_ColorConvert_IUV_RGBX(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImgU = node->paramList[0];
		AgoData * oImgV = node->paramList[1];
		AgoData * iImg = node->paramList[2];
		if (HafCpu_ColorConvert_IUV_RGBX(oImgU->u.img.width, oImgU->u.img.height, oImgU->buffer, oImgU->u.img.stride_in_bytes, 
										 oImgV->buffer, oImgV->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[2]->u.img.width;
		vx_uint32 height = node->paramList[2]->u.img.height;
		if (node->paramList[2]->u.img.format != VX_DF_IMAGE_RGBX)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || (width & 1) || (height & 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes and format
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width >> 1;
		meta->data.u.img.height = height >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		meta = &node->metaList[1];
		meta->data.u.img.width = width >> 1;
		meta->data.u.img.height = height >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ColorConvert(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
#if ENABLE_OPENCL		
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif			
			;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out1 = node->paramList[0];
		AgoData * out2 = node->paramList[1];
		AgoData * inp = node->paramList[2];
		out1->u.img.rect_valid.start_x = (inp->u.img.rect_valid.start_x + 1) >> 1;
		out1->u.img.rect_valid.start_y = (inp->u.img.rect_valid.start_y + 1) >> 1;
		out1->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x >> 1;
		out1->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y >> 1;
		out2->u.img.rect_valid.start_x = (inp->u.img.rect_valid.start_x + 1) >> 1;
		out2->u.img.rect_valid.start_y = (inp->u.img.rect_valid.start_y + 1) >> 1;
		out2->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x >> 1;
		out2->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y >> 1;
	}
	return status;
}

int agoKernel_ColorConvert_UV12_RGB(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ColorConvert_UV12_RGB(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_RGB)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || (width & 1) || (height & 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes and format
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width >> 1;
		meta->data.u.img.height = height >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U16;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ColorConvert(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
#if ENABLE_OPENCL		
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif			
			;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_ColorConvert_UV12_RGBX(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ColorConvert_UV12_RGBX(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_RGBX)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height || (width & 1) || (height & 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes and format
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width >> 1;
		meta->data.u.img.height = height >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U16;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ColorConvert(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
#if ENABLE_OPENCL		
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif			
			;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_Box_U8_U8_3x3(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_Box_U8_U8_3x3(oImg->u.img.width, oImg->u.img.height - 2, oImg->buffer + oImg->u.img.stride_in_bytes, oImg->u.img.stride_in_bytes,
			iImg->buffer + iImg->u.img.stride_in_bytes, iImg->u.img.stride_in_bytes, node->localDataPtr)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, true, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		int alignedWidth = (node->paramList[0]->u.img.width + 15) & ~15;		// Next highest multiple of 16, so that the buffer is aligned for all three lines
		node->localDataSize = 3 * alignedWidth * sizeof(vx_uint16);				// Three rows (+some extra) worth of scratch memory			
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		// re-use LinearFilter_ANY_U8
		float filterCoef[] = { 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f };
		AgoData filter;
		filter.ref.type = VX_TYPE_MATRIX; filter.u.mat.type = VX_TYPE_FLOAT32; filter.u.mat.columns = filter.u.mat.rows = 3; filter.buffer = (vx_uint8 *)filterCoef; filter.ref.read_only = true;
		status = HafGpu_LinearFilter_ANY_U8(node, VX_DF_IMAGE_U8, &filter, false);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_Dilate_U8_U8_3x3(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_Dilate_U8_U8_3x3(oImg->u.img.width, oImg->u.img.height - 2, oImg->buffer + oImg->u.img.stride_in_bytes, oImg->u.img.stride_in_bytes,
			iImg->buffer + iImg->u.img.stride_in_bytes, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, true, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_NonLinearFilter_3x3_ANY_U8(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_Erode_U8_U8_3x3(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_Erode_U8_U8_3x3(oImg->u.img.width, oImg->u.img.height - 2, oImg->buffer + oImg->u.img.stride_in_bytes, oImg->u.img.stride_in_bytes,
			iImg->buffer + iImg->u.img.stride_in_bytes, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, true, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_NonLinearFilter_3x3_ANY_U8(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_Median_U8_U8_3x3(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_Median_U8_U8_3x3(oImg->u.img.width, oImg->u.img.height - 2, oImg->buffer + oImg->u.img.stride_in_bytes, oImg->u.img.stride_in_bytes, 
			iImg->buffer + iImg->u.img.stride_in_bytes, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, true, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_NonLinearFilter_3x3_ANY_U8(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_Gaussian_U8_U8_3x3(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_Gaussian_U8_U8_3x3(oImg->u.img.width, oImg->u.img.height - 2, oImg->buffer + oImg->u.img.stride_in_bytes, oImg->u.img.stride_in_bytes,
			iImg->buffer + iImg->u.img.stride_in_bytes, iImg->u.img.stride_in_bytes, node->localDataPtr)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, true, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		int alignedWidth = (node->paramList[0]->u.img.width + 15) & ~15;		// Next highest multiple of 16, so that the buffer is aligned for all three lines
		node->localDataSize = 3 * alignedWidth * sizeof(vx_uint16);				// Three rows (+some extra) worth of scratch memory			
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		// re-use LinearFilter_ANY_U8
		float filterCoef[] = { 0.0625f, 0.125f, 0.0625f, 0.125f, 0.25f, 0.125f, 0.0625f, 0.125f, 0.0625f };
		AgoData filter;
		filter.ref.type = VX_TYPE_MATRIX; filter.u.mat.type = VX_TYPE_FLOAT32; filter.u.mat.columns = filter.u.mat.rows = 3; filter.buffer = (vx_uint8 *)filterCoef; filter.ref.read_only = true;
		status = HafGpu_LinearFilter_ANY_U8(node, VX_DF_IMAGE_U8, &filter, false);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_ScaleGaussianHalf_U8_U8_3x3(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ScaleGaussianHalf_U8_U8_3x3(oImg->u.img.width, oImg->u.img.height - 2, oImg->buffer + oImg->u.img.stride_in_bytes, oImg->u.img.stride_in_bytes,
			iImg->buffer + iImg->u.img.stride_in_bytes, iImg->u.img.stride_in_bytes, node->localDataPtr)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes and format
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = (width + 1) >> 1;
		meta->data.u.img.height = (height + 1) >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		int alignedWidth = (node->paramList[0]->u.img.width + 15) & ~15;		// Next highest multiple of 16, so that the buffer is aligned for all three lines
		node->localDataSize = (2 * alignedWidth * sizeof(vx_uint16))+16;				// 2 rows (+some extra) worth of scratch memory			
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ScaleGaussianHalf(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(((inp->u.img.rect_valid.start_x + 1) >> 1) + 1, width);
		out->u.img.rect_valid.start_y = min(((inp->u.img.rect_valid.start_y + 1) >> 1) + 1, height);
		out->u.img.rect_valid.end_x = max((int)((inp->u.img.rect_valid.end_x + 1) >> 1) - 1, 0);
		out->u.img.rect_valid.end_y = max((int)((inp->u.img.rect_valid.end_y + 1) >> 1) - 1, 0);
	}
	return status;
}

int agoKernel_ScaleGaussianHalf_U8_U8_5x5(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		bool sampleFirstRow = (iImg->u.img.height & 1) ? true : false;
		bool sampleFirstColumn = (iImg->u.img.width & 1) ? true : false;
		if (iImg->u.img.width < 5 || iImg->u.img.height < 5 || oImg->u.img.width < 3 || oImg->u.img.height < 3) {
			status = VX_ERROR_INVALID_DIMENSION;
		}
		else if (HafCpu_ScaleGaussianHalf_U8_U8_5x5(oImg->u.img.width, oImg->u.img.height - 2, oImg->buffer + oImg->u.img.stride_in_bytes, oImg->u.img.stride_in_bytes, 
			iImg->buffer + (2 * iImg->u.img.stride_in_bytes), iImg->u.img.stride_in_bytes, sampleFirstRow, sampleFirstColumn, node->localDataPtr)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes and format
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = (width + 1) >> 1;
		meta->data.u.img.height = (height + 1) >> 1;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		int alignedDstStride = (node->paramList[0]->u.img.stride_in_bytes + 15) & ~15;
		node->localDataSize = 5 * 2 * alignedDstStride * sizeof(vx_int16);
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ScaleGaussianHalf(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(((inp->u.img.rect_valid.start_x + 1) >> 1) + 1, width);
		out->u.img.rect_valid.start_y = min(((inp->u.img.rect_valid.start_y + 1) >> 1) + 1, height);
		out->u.img.rect_valid.end_x = max((int)((inp->u.img.rect_valid.end_x + 1) >> 1) - 1, 0);
		out->u.img.rect_valid.end_y = max((int)((inp->u.img.rect_valid.end_y + 1) >> 1) - 1, 0);
	}
	return status;
}

int agoKernel_ScaleGaussianOrb_U8_U8_5x5(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ScaleGaussianOrb_U8_U8_5x5(oImg->u.img.width, oImg->u.img.height - 4, oImg->buffer + (2 * oImg->u.img.stride_in_bytes), oImg->u.img.stride_in_bytes, 
			iImg->buffer, iImg->u.img.stride_in_bytes, iImg->u.img.width, iImg->u.img.height, node->localDataPtr)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes and format
		vx_uint32 dwidth = (vx_uint32)ceilf(VX_SCALE_PYRAMID_ORB * width);
		vx_uint32 dheight = (vx_uint32)ceilf(VX_SCALE_PYRAMID_ORB * height);
		if ((node->paramList[0]->u.img.width && abs((int)dwidth - (int)node->paramList[0]->u.img.width) > 1) ||
			(node->paramList[0]->u.img.height && abs((int)dheight - (int)node->paramList[0]->u.img.height) > 1))
			return VX_ERROR_INVALID_DIMENSION;
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = node->paramList[0]->u.img.width ? node->paramList[0]->u.img.width : dwidth;
		meta->data.u.img.height = node->paramList[0]->u.img.height ? node->paramList[0]->u.img.height : dheight;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		int alignedWidth = (node->paramList[0]->u.img.width + 15) & ~15;		// Next highest multiple of 16, so that the buffer is aligned for all three lines
		node->localDataSize = (3 * node->paramList[1]->u.img.width)+(2 * alignedWidth) + 128;				// 2 rows (+some extra) worth of scratch memory			
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_ScaleGaussianOrb(node, VX_INTERPOLATION_TYPE_NEAREST_NEIGHBOR);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(((vx_uint32)ceilf(VX_SCALE_PYRAMID_ORB * inp->u.img.rect_valid.start_x)) + 1, width);
		out->u.img.rect_valid.start_y = min(((vx_uint32)ceilf(VX_SCALE_PYRAMID_ORB * inp->u.img.rect_valid.start_y)) + 1, height);
		out->u.img.rect_valid.end_x = max(((int)floorf(VX_SCALE_PYRAMID_ORB * inp->u.img.rect_valid.end_x)) - 1, 0);
		out->u.img.rect_valid.end_y = max(((int)floorf(VX_SCALE_PYRAMID_ORB * inp->u.img.rect_valid.end_y)) - 1, 0);
	}
	return status;
}

int agoKernel_Convolve_U8_U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iConv = node->paramList[2];
		vx_uint32 convolutionWidth = (vx_uint32)iConv->u.conv.columns;
		vx_uint32 convolutionHeight = (vx_uint32)iConv->u.conv.rows;
		if (convolutionWidth == 3) {
			status = HafCpu_Convolve_U8_U8_3xN(oImg->u.img.width, oImg->u.img.height - convolutionHeight + 1,
				oImg->buffer + oImg->u.img.stride_in_bytes * (convolutionHeight >> 1), oImg->u.img.stride_in_bytes,
				iImg->buffer + iImg->u.img.stride_in_bytes * (convolutionHeight >> 1), iImg->u.img.stride_in_bytes, (vx_int16 *)iConv->buffer, convolutionHeight, iConv->u.conv.shift);
		}
		else if (convolutionWidth == 5) {
			status = HafCpu_Convolve_U8_U8_5xN(oImg->u.img.width, oImg->u.img.height - convolutionHeight + 1,
				oImg->buffer + oImg->u.img.stride_in_bytes * (convolutionHeight >> 1), oImg->u.img.stride_in_bytes,
				iImg->buffer + iImg->u.img.stride_in_bytes * (convolutionHeight >> 1), iImg->u.img.stride_in_bytes, (vx_int16 *)iConv->buffer, convolutionHeight, iConv->u.conv.shift);
		}
		else if (convolutionWidth == 7) {
			status = HafCpu_Convolve_U8_U8_7xN(oImg->u.img.width, oImg->u.img.height - convolutionHeight + 1,
				oImg->buffer + oImg->u.img.stride_in_bytes * (convolutionHeight >> 1), oImg->u.img.stride_in_bytes,
				iImg->buffer + iImg->u.img.stride_in_bytes * (convolutionHeight >> 1), iImg->u.img.stride_in_bytes, (vx_int16 *)iConv->buffer, convolutionHeight, iConv->u.conv.shift);
		}
		else if (convolutionWidth == 9) {
			status = HafCpu_Convolve_U8_U8_9xN(oImg->u.img.width, oImg->u.img.height - convolutionHeight + 1,
				oImg->buffer + oImg->u.img.stride_in_bytes * (convolutionHeight >> 1), oImg->u.img.stride_in_bytes,
				iImg->buffer + iImg->u.img.stride_in_bytes * (convolutionHeight >> 1), iImg->u.img.stride_in_bytes, (vx_int16 *)iConv->buffer, convolutionHeight, iConv->u.conv.shift);
		}
		else {
			status = HafCpu_Convolve_U8_U8_MxN(oImg->u.img.width, oImg->u.img.height - convolutionHeight + 1,
				oImg->buffer + oImg->u.img.stride_in_bytes * (convolutionHeight >> 1), oImg->u.img.stride_in_bytes,
				iImg->buffer + iImg->u.img.stride_in_bytes * (convolutionHeight >> 1), iImg->u.img.stride_in_bytes, (vx_int16 *)iConv->buffer, convolutionWidth, convolutionHeight, iConv->u.conv.shift);
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		int M = (int) node->paramList[2]->u.conv.columns >> 1;
		int N = (int) node->paramList[2]->u.conv.rows >> 1;
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (!(node->paramList[2]->u.conv.rows & 1) || !(node->paramList[2]->u.conv.columns & 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes and format
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_U8;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_LinearFilter_ANY_U8(node, VX_DF_IMAGE_U8, node->paramList[2], false);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		int M = (int) node->paramList[2]->u.conv.columns >> 1;
		int N = (int) node->paramList[2]->u.conv.rows >> 1;
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + M, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + N, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - M, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - N, 0);
	}
	return status;
}

int agoKernel_Convolve_S16_U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iConv = node->paramList[2];
		vx_uint32 convolutionWidth = (vx_uint32)iConv->u.conv.columns;
		vx_uint32 convolutionHeight = (vx_uint32)iConv->u.conv.rows;
		if (convolutionWidth == 3) {
			status = HafCpu_Convolve_S16_U8_3xN(oImg->u.img.width, oImg->u.img.height - convolutionHeight + 1,
				(vx_int16 *)(oImg->buffer + oImg->u.img.stride_in_bytes * (convolutionHeight >> 1)), oImg->u.img.stride_in_bytes,
				iImg->buffer + iImg->u.img.stride_in_bytes * (convolutionHeight >> 1), iImg->u.img.stride_in_bytes, (vx_int16 *)iConv->buffer, convolutionHeight, iConv->u.conv.shift);
		}
		else if (convolutionWidth == 5) {
			status = HafCpu_Convolve_S16_U8_5xN(oImg->u.img.width, oImg->u.img.height - convolutionHeight + 1,
				(vx_int16 *)(oImg->buffer + oImg->u.img.stride_in_bytes * (convolutionHeight >> 1)), oImg->u.img.stride_in_bytes,
				iImg->buffer + iImg->u.img.stride_in_bytes * (convolutionHeight >> 1), iImg->u.img.stride_in_bytes, (vx_int16 *)iConv->buffer, convolutionHeight, iConv->u.conv.shift);
		}
		else if (convolutionWidth == 7) {
			status = HafCpu_Convolve_S16_U8_7xN(oImg->u.img.width, oImg->u.img.height - convolutionHeight + 1,
				(vx_int16 *)(oImg->buffer + oImg->u.img.stride_in_bytes * (convolutionHeight >> 1)), oImg->u.img.stride_in_bytes,
				iImg->buffer + iImg->u.img.stride_in_bytes * (convolutionHeight >> 1), iImg->u.img.stride_in_bytes, (vx_int16 *)iConv->buffer, convolutionHeight, iConv->u.conv.shift);
		}
		else if (convolutionWidth == 9) {
			status = HafCpu_Convolve_S16_U8_9xN(oImg->u.img.width, oImg->u.img.height - convolutionHeight + 1,
				(vx_int16 *)(oImg->buffer + oImg->u.img.stride_in_bytes * (convolutionHeight >> 1)), oImg->u.img.stride_in_bytes,
				iImg->buffer + iImg->u.img.stride_in_bytes * (convolutionHeight >> 1), iImg->u.img.stride_in_bytes, (vx_int16 *)iConv->buffer, convolutionHeight, iConv->u.conv.shift);
		}
		else {
			status = HafCpu_Convolve_S16_U8_MxN(oImg->u.img.width, oImg->u.img.height - convolutionHeight + 1,
				(vx_int16 *)(oImg->buffer + oImg->u.img.stride_in_bytes * (convolutionHeight >> 1)), oImg->u.img.stride_in_bytes,
				iImg->buffer + iImg->u.img.stride_in_bytes * (convolutionHeight >> 1), iImg->u.img.stride_in_bytes, (vx_int16 *)iConv->buffer, convolutionWidth, convolutionHeight, iConv->u.conv.shift);
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		int M = (int) node->paramList[2]->u.conv.columns >> 1;
		int N = (int) node->paramList[2]->u.conv.rows >> 1;
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (!(node->paramList[2]->u.conv.rows & 1) || !(node->paramList[2]->u.conv.columns & 1))
			return VX_ERROR_INVALID_DIMENSION;
		// set output image sizes and format
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = VX_DF_IMAGE_S16;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_LinearFilter_ANY_U8(node, VX_DF_IMAGE_S16, node->paramList[2], false);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		int M = (int) node->paramList[2]->u.conv.columns >> 1;
		int N = (int) node->paramList[2]->u.conv.rows >> 1;
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + M, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + N, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - M, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - N, 0);
	}
	return status;
}

int agoKernel_LinearFilter_ANY_ANY(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		if (node->paramList[0]->u.img.format == VX_DF_IMAGE_U8 && node->paramList[1]->u.img.format == VX_DF_IMAGE_U8) {
			status = agoKernel_Convolve_U8_U8(node, cmd);
		}
		else if (node->paramList[0]->u.img.format == VX_DF_IMAGE_S16 && node->paramList[1]->u.img.format == VX_DF_IMAGE_U8) {
			status = agoKernel_Convolve_S16_U8(node, cmd);
		}
		else {
			// TBD: not implemented yet
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[1]->u.img.width;
		vx_uint32 height = node->paramList[1]->u.img.height;
		int M = (int) node->paramList[2]->u.mat.columns >> 1;
		int N = (int) node->paramList[2]->u.mat.rows >> 1;
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_U8 &&
			node->paramList[1]->u.img.format != VX_DF_IMAGE_S16 &&
			node->paramList[1]->u.img.format != VX_DF_IMAGE_F32_AMD)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (!(node->paramList[2]->u.mat.rows & 1) || !(node->paramList[2]->u.mat.columns & 1))
			return VX_ERROR_INVALID_DIMENSION;
		else if (node->paramList[2]->u.mat.type != VX_TYPE_FLOAT32)
			return VX_ERROR_INVALID_FORMAT;
		else if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8 &&
			node->paramList[0]->u.img.format != VX_DF_IMAGE_S16 &&
			node->paramList[0]->u.img.format != VX_DF_IMAGE_F32_AMD)
			return VX_ERROR_INVALID_FORMAT;
		// set output image sizes and format
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = node->paramList[0]->u.img.format;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		if (node->paramList[1]->u.img.format == VX_DF_IMAGE_U8) {
			status = HafGpu_LinearFilter_ANY_U8(node, node->paramList[0]->u.img.format, node->paramList[2], true);
		}
		else if (node->paramList[1]->u.img.format == VX_DF_IMAGE_S16) {
			status = HafGpu_LinearFilter_ANY_S16(node, node->paramList[0]->u.img.format, node->paramList[2], true);
		}
		else if (node->paramList[1]->u.img.format == VX_DF_IMAGE_F32_AMD) {
			status = HafGpu_LinearFilter_ANY_F32(node, node->paramList[0]->u.img.format, node->paramList[2], true);
		}
		else {
			// TBD: not implemented yet
		}
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		int M = (int) node->paramList[2]->u.mat.columns >> 1;
		int N = (int) node->paramList[2]->u.mat.rows >> 1;
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + M, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + N, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - M, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - N, 0);
	}
	return status;
}

int agoKernel_LinearFilter_ANYx2_ANY(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		// TBD: not implemented yet
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[2]->u.img.width;
		vx_uint32 height = node->paramList[2]->u.img.height;
		int M = (int) node->paramList[3]->u.mat.columns >> 1;
		int N = (int) node->paramList[3]->u.mat.rows >> 1;
		if (node->paramList[2]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (!(node->paramList[3]->u.mat.rows & 1) || !(node->paramList[3]->u.mat.columns & 1) ||
			      (node->paramList[3]->u.mat.rows != node->paramList[4]->u.mat.rows) ||
				  (node->paramList[3]->u.mat.columns != node->paramList[4]->u.mat.columns))
			return VX_ERROR_INVALID_DIMENSION;
		else if ((node->paramList[3]->u.mat.type != VX_TYPE_FLOAT32) || (node->paramList[3]->u.mat.type != node->paramList[4]->u.mat.type))
			return VX_ERROR_INVALID_FORMAT;
		else if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8 &&
			node->paramList[0]->u.img.format != VX_DF_IMAGE_S16 &&
			node->paramList[0]->u.img.format != VX_DF_IMAGE_F32_AMD &&
			node->paramList[0]->u.img.format != node->paramList[1]->u.img.format)
			return VX_ERROR_INVALID_FORMAT;
		// set output image sizes and format
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = node->paramList[0]->u.img.format;
		meta = &node->metaList[1];
		meta->data.u.img.width = width;
		meta->data.u.img.height = height;
		meta->data.u.img.format = node->paramList[0]->u.img.format;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		if (node->paramList[2]->u.img.format == VX_DF_IMAGE_U8) {
			status = HafGpu_LinearFilter_ANYx2_U8(node, node->paramList[0]->u.img.format, node->paramList[3], node->paramList[4], true);
		}
		else {
			// TBD: not implemented yet
		}
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
#if ENABLE_OPENCL        
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out1 = node->paramList[0];
		AgoData * out2 = node->paramList[1];
		AgoData * inp = node->paramList[2];
		int M = (int) node->paramList[3]->u.mat.columns >> 1;
		int N = (int) node->paramList[3]->u.mat.rows >> 1;
		vx_uint32 width = out1->u.img.width;
		vx_uint32 height = out1->u.img.height;
		out1->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + M, width);
		out1->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + N, height);
		out1->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - M, 0);
		out1->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - N, 0);
		out2->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + M, width);
		out2->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + N, height);
		out2->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - M, 0);
		out2->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - N, 0);
	}
	return status;
}

int agoKernel_SobelMagnitude_S16_U8_3x3(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_SobelMagnitude_S16_U8_3x3(oImg->u.img.width, oImg->u.img.height - 2, (vx_int16 *)(oImg->buffer + oImg->u.img.stride_in_bytes), oImg->u.img.stride_in_bytes, 
			iImg->buffer + iImg->u.img.stride_in_bytes, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8, true, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_SobelSpecialCases(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_SobelPhase_U8_U8_3x3(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_SobelPhase_U8_U8_3x3(oImg->u.img.width, oImg->u.img.height - 2, oImg->buffer + oImg->u.img.stride_in_bytes, oImg->u.img.stride_in_bytes, 
			iImg->buffer + iImg->u.img.stride_in_bytes, iImg->u.img.stride_in_bytes, node->localDataPtr)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, true, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		int alignedWidth = (node->paramList[0]->u.img.width + 15) & ~15;	// Next highest multiple of 16, so that the buffer is aligned for all three lines
		int alignedStride = (node->paramList[0]->u.img.stride_in_bytes + 15) & ~15;
		node->localDataSize = (alignedStride * node->paramList[0]->u.img.height * sizeof(vx_int16) * 2) + (6 * alignedWidth * sizeof(vx_int16));	// Two buffers for Gx and Gy and Three rows (+some extra) worth of scratch memory
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_SobelSpecialCases(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_SobelMagnitudePhase_S16U8_U8_3x3(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg1 = node->paramList[0];
		AgoData * oImg2 = node->paramList[1];
		AgoData * iImg = node->paramList[2];
		if (HafCpu_SobelMagnitudePhase_S16U8_U8_3x3(oImg1->u.img.width, oImg1->u.img.height - 2, 
			(vx_int16 *)(oImg1->buffer + oImg1->u.img.stride_in_bytes), oImg1->u.img.stride_in_bytes,
			oImg2->buffer + oImg2->u.img.stride_in_bytes, oImg2->u.img.stride_in_bytes, 
			iImg->buffer + iImg->u.img.stride_in_bytes, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_2OUT_1IN(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, true, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_SobelSpecialCases(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out1 = node->paramList[0];
		AgoData * out2 = node->paramList[1];
		AgoData * inp = node->paramList[2];
		vx_uint32 width = inp->u.img.width;
		vx_uint32 height = inp->u.img.height;
		out1->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out1->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out1->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out1->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
		out2->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out2->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out2->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out2->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_Sobel_S16S16_U8_3x3_GXY(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg1 = node->paramList[0];
		AgoData * oImg2 = node->paramList[1];
		AgoData * iImg = node->paramList[2];
		if (HafCpu_Sobel_S16S16_U8_3x3_GXY(oImg1->u.img.width, oImg1->u.img.height - 2, 
			(vx_int16 *)(oImg1->buffer + oImg1->u.img.stride_in_bytes), oImg1->u.img.stride_in_bytes,
			(vx_int16 *)(oImg2->buffer + oImg2->u.img.stride_in_bytes), oImg2->u.img.stride_in_bytes,
			iImg->buffer + iImg->u.img.stride_in_bytes, iImg->u.img.stride_in_bytes, node->localDataPtr))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_2OUT_1IN(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8, true, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		int alignedWidth = (node->paramList[0]->u.img.width + 15) & ~15;		// Next highest multiple of 16, so that the buffer is aligned for all three lines
		node->localDataSize = 6 * alignedWidth * sizeof(vx_int16);				// Three rows (+some extra) worth of scratch memory	- each row is Gx and Gy	
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_SobelSpecialCases(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out1 = node->paramList[0];
		AgoData * out2 = node->paramList[1];
		AgoData * inp = node->paramList[2];
		vx_uint32 width = inp->u.img.width;
		vx_uint32 height = inp->u.img.height;
		out1->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out1->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out1->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out1->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
		out2->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out2->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out2->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out2->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_Sobel_S16_U8_3x3_GX(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_Sobel_S16_U8_3x3_GX(oImg->u.img.width, oImg->u.img.height - 2, (vx_int16 *)(oImg->buffer + oImg->u.img.stride_in_bytes), oImg->u.img.stride_in_bytes, 
			iImg->buffer + iImg->u.img.stride_in_bytes, iImg->u.img.stride_in_bytes, node->localDataPtr)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8, true, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		int alignedWidth = (node->paramList[0]->u.img.width + 15) & ~15;		// Next highest multiple of 16, so that the buffer is aligned for all three lines
		node->localDataSize = 3 * alignedWidth * sizeof(vx_int16);				// Three rows (+some extra) worth of scratch memory			
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_SobelSpecialCases(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_Sobel_S16_U8_3x3_GY(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_Sobel_S16_U8_3x3_GY(oImg->u.img.width, oImg->u.img.height - 2, (vx_int16 *)(oImg->buffer + oImg->u.img.stride_in_bytes), oImg->u.img.stride_in_bytes, 
			iImg->buffer + iImg->u.img.stride_in_bytes, iImg->u.img.stride_in_bytes, node->localDataPtr)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8, true, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		int alignedWidth = (node->paramList[0]->u.img.width + 15) & ~15;		// Next highest multiple of 16, so that the buffer is aligned for all three lines
		node->localDataSize = 3 * alignedWidth * sizeof(vx_int16);				// Three rows (+some extra) worth of scratch memory			
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_SobelSpecialCases(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_Dilate_U1_U8_3x3(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_Dilate_U1_U8_3x3(oImg->u.img.width, oImg->u.img.height - 2, oImg->buffer + oImg->u.img.stride_in_bytes, oImg->u.img.stride_in_bytes, 
			iImg->buffer + iImg->u.img.stride_in_bytes, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8, true, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_NonLinearFilter_3x3_ANY_U8(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_Erode_U1_U8_3x3(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_Erode_U1_U8_3x3(oImg->u.img.width, oImg->u.img.height - 2, oImg->buffer + oImg->u.img.stride_in_bytes, oImg->u.img.stride_in_bytes, 
			iImg->buffer + iImg->u.img.stride_in_bytes, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U8, true, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_NonLinearFilter_3x3_ANY_U8(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_Dilate_U1_U1_3x3(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_Dilate_U1_U1_3x3(oImg->u.img.width, oImg->u.img.height - 2, oImg->buffer + oImg->u.img.stride_in_bytes, oImg->u.img.stride_in_bytes, 
			iImg->buffer + iImg->u.img.stride_in_bytes, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD, true, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_NonLinearFilter_3x3_ANY_U1(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_Erode_U1_U1_3x3(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_Erode_U1_U1_3x3(oImg->u.img.width, oImg->u.img.height - 2, oImg->buffer + oImg->u.img.stride_in_bytes, oImg->u.img.stride_in_bytes, 
			iImg->buffer + iImg->u.img.stride_in_bytes, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U1_AMD, VX_DF_IMAGE_U1_AMD, true, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_NonLinearFilter_3x3_ANY_U1(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_Dilate_U8_U1_3x3(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_Dilate_U8_U1_3x3(oImg->u.img.width, oImg->u.img.height - 2, oImg->buffer + oImg->u.img.stride_in_bytes, oImg->u.img.stride_in_bytes, 
			iImg->buffer + iImg->u.img.stride_in_bytes, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD, true, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_NonLinearFilter_3x3_ANY_U1(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_Erode_U8_U1_3x3(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_Erode_U8_U1_3x3(oImg->u.img.width, oImg->u.img.height - 2, oImg->buffer + oImg->u.img.stride_in_bytes, oImg->u.img.stride_in_bytes, 
			iImg->buffer + iImg->u.img.stride_in_bytes, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U1_AMD, true, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_NonLinearFilter_3x3_ANY_U1(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_FastCorners_XY_U8_Supression(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oXY = node->paramList[0];
		AgoData * oNumCorners = node->paramList[1];
		AgoData * iImg = node->paramList[2];
		vx_float32 strength_threshold = node->paramList[3]->u.scalar.u.f;
		vx_uint32 numXY = 0;
		if (HafCpu_FastCorners_XY_U8_Supression((vx_uint32)oXY->u.arr.capacity, (vx_keypoint_t *)oXY->buffer, &numXY, 
			iImg->u.img.width, iImg->u.img.height, iImg->buffer, iImg->u.img.stride_in_bytes, strength_threshold, node->localDataPtr)) {
			status = VX_FAILURE;
		}
		else {
			oXY->u.arr.numitems = min(numXY, (vx_uint32)oXY->u.arr.capacity);
			if (oNumCorners) oNumCorners->u.scalar.u.s = numXY;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[2]->u.img.width;
		vx_uint32 height = node->paramList[2]->u.img.height;
		if (node->paramList[2]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (node->paramList[3]->u.scalar.type != VX_TYPE_FLOAT32)
			return VX_ERROR_INVALID_TYPE;
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.arr.itemtype = VX_TYPE_KEYPOINT;
		meta = &node->metaList[1];
		meta->data.u.scalar.type = VX_TYPE_SIZE;
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_FastCorners_XY_U8(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_initialize) {
		node->localDataSize = node->paramList[2]->u.img.width * node->paramList[2]->u.img.height;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_FastCorners_XY_U8_NoSupression(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oXY = node->paramList[0];
		AgoData * oNumCorners = node->paramList[1];
		AgoData * iImg = node->paramList[2];
		vx_float32 strength_threshold = node->paramList[3]->u.scalar.u.f;
		vx_uint32 numXY = 0;
		if (HafCpu_FastCorners_XY_U8_NoSupression((vx_uint32)oXY->u.arr.capacity, (vx_keypoint_t *)oXY->buffer, &numXY,
			iImg->u.img.width, iImg->u.img.height, iImg->buffer, iImg->u.img.stride_in_bytes, strength_threshold)) {
			status = VX_FAILURE;
		}
		else {
			oXY->u.arr.numitems = min(numXY, (vx_uint32)oXY->u.arr.capacity);
			if (oNumCorners) oNumCorners->u.scalar.u.s = numXY;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		vx_uint32 width = node->paramList[2]->u.img.width;
		vx_uint32 height = node->paramList[2]->u.img.height;
		if (node->paramList[2]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!width || !height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (node->paramList[3]->u.scalar.type != VX_TYPE_FLOAT32)
			return VX_ERROR_INVALID_TYPE;
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.arr.itemtype = VX_TYPE_KEYPOINT;
		meta = &node->metaList[1];
		meta->data.u.scalar.type = VX_TYPE_SIZE;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_FastCorners_XY_U8(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_HarrisSobel_HG3_U8_3x3(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_HarrisSobel_HG3_U8_3x3(oImg->u.img.width, oImg->u.img.height, (vx_float32 *)oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes, node->localDataPtr)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_F32x3_AMD, VX_DF_IMAGE_U8, true, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		int alignedWidth = (node->paramList[0]->u.img.width + 15) & ~15;		// Next highest multiple of 16, so that the buffer is aligned for all three lines
		node->localDataSize = 6 * alignedWidth * sizeof(vx_int16);				// Three rows (one vx_int16 for Gx and one for Gy + some extra) worth of scratch memory			
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_HarrisSobelFilters(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_HarrisSobel_HG3_U8_5x5(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_HarrisSobel_HG3_U8_5x5(oImg->u.img.width, oImg->u.img.height, (vx_float32 *)oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes, node->localDataPtr)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_F32x3_AMD, VX_DF_IMAGE_U8, true, 2, 2);
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		int alignedWidth = (node->paramList[0]->u.img.width + 15) & ~15;		// Next highest multiple of 16, so that the buffer is aligned for all three lines
		node->localDataSize = 10 * alignedWidth * sizeof(vx_int16);				// Five rows (one vx_int16 for Gx and one for Gy + some extra) worth of scratch memory		
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_HarrisSobelFilters(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 2, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 2, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 2, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 2, 0);
	}
	return status;
}

int agoKernel_HarrisSobel_HG3_U8_7x7(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_HarrisSobel_HG3_U8_7x7(oImg->u.img.width, oImg->u.img.height, (vx_float32 *)oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes, node->localDataPtr)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_F32x3_AMD, VX_DF_IMAGE_U8, true, 3, 3);
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		int alignedWidth = (node->paramList[0]->u.img.width + 15) & ~15;		// Next highest multiple of 16, so that the buffer is aligned for all three lines
		node->localDataSize = 14 * alignedWidth * sizeof(vx_int16);				// Seven rows (one vx_int16 for Gx and one for Gy + some extra) worth of scratch memory			
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_HarrisSobelFilters(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 3, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 3, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 3, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 3, 0);
	}
	return status;
}

int agoKernel_HarrisScore_HVC_HG3_3x3(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		vx_float32 sensitivity = node->paramList[2]->u.scalar.u.f;
		vx_int32 gradient_size = node->paramList[4]->u.scalar.u.i;
		vx_float32 strength_threshold = node->paramList[3]->u.scalar.u.f;
		vx_float32 normFactor = 255.0f * (1 << (gradient_size - 1)) * 3;
		normFactor = normFactor * normFactor * normFactor * normFactor;
		if (HafCpu_HarrisScore_HVC_HG3_3x3(oImg->u.img.width, oImg->u.img.height, (vx_float32 *)oImg->buffer, oImg->u.img.stride_in_bytes,
			(vx_float32 *)iImg->buffer, iImg->u.img.stride_in_bytes, sensitivity, strength_threshold, normFactor)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN_3S(node, VX_DF_IMAGE_F32_AMD, VX_DF_IMAGE_F32x3_AMD, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, VX_TYPE_INT32, true, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_HarrisScoreFilters(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_HarrisScore_HVC_HG3_5x5(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		vx_float32 sensitivity = node->paramList[2]->u.scalar.u.f;
		vx_int32 gradient_size = node->paramList[4]->u.scalar.u.i;
		vx_float32 strength_threshold = node->paramList[3]->u.scalar.u.f;
		vx_float32 normFactor = 255.0f * (1 << (gradient_size - 1)) * 5;
		normFactor = normFactor * normFactor * normFactor * normFactor;
		if (HafCpu_HarrisScore_HVC_HG3_5x5(oImg->u.img.width, oImg->u.img.height, (vx_float32 *)oImg->buffer, oImg->u.img.stride_in_bytes,
			(vx_float32 *)iImg->buffer, iImg->u.img.stride_in_bytes, sensitivity, strength_threshold, normFactor)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN_3S(node, VX_DF_IMAGE_F32_AMD, VX_DF_IMAGE_F32x3_AMD, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, VX_TYPE_INT32, true, 2, 2);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_HarrisScoreFilters(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 2, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 2, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 2, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 2, 0);
	}
	return status;
}

int agoKernel_HarrisScore_HVC_HG3_7x7(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		vx_float32 sensitivity = node->paramList[2]->u.scalar.u.f;
		vx_int32 gradient_size = node->paramList[4]->u.scalar.u.i;
		vx_float32 strength_threshold = node->paramList[3]->u.scalar.u.f;
		vx_float32 normFactor = 255.0f * (1 << (gradient_size - 1)) * 7;
		normFactor = normFactor * normFactor * normFactor * normFactor;
		if (HafCpu_HarrisScore_HVC_HG3_7x7(oImg->u.img.width, oImg->u.img.height, (vx_float32 *)oImg->buffer, oImg->u.img.stride_in_bytes,
			(vx_float32 *)iImg->buffer, iImg->u.img.stride_in_bytes, sensitivity, strength_threshold, normFactor)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN_3S(node, VX_DF_IMAGE_F32_AMD, VX_DF_IMAGE_F32x3_AMD, VX_TYPE_FLOAT32, VX_TYPE_FLOAT32, VX_TYPE_INT32, true, 3, 3);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_HarrisScoreFilters(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 3, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 3, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 3, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 3, 0);
	}
	return status;
}

int agoKernel_CannySobelSuppThreshold_U8_U8_3x3_L1NORM(AgoNode * node, AgoKernelCommand cmd)
{
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
    else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_CannySuppThreshold_U8(node, VX_DF_IMAGE_U8, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_CannySobelSuppThreshold_U8_U8_3x3_L2NORM(AgoNode * node, AgoKernelCommand cmd)
{
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_CannySuppThreshold_U8(node, VX_DF_IMAGE_U8, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_CannySobelSuppThreshold_U8_U8_5x5_L1NORM(AgoNode * node, AgoKernelCommand cmd)
{
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_CannySuppThreshold_U8(node, VX_DF_IMAGE_U8, 2, 2);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_CannySobelSuppThreshold_U8_U8_5x5_L2NORM(AgoNode * node, AgoKernelCommand cmd)
{
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_CannySuppThreshold_U8(node, VX_DF_IMAGE_U8, 2, 2);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_CannySobelSuppThreshold_U8_U8_7x7_L1NORM(AgoNode * node, AgoKernelCommand cmd)
{
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_CannySuppThreshold_U8(node, VX_DF_IMAGE_U8, 3, 3);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_CannySobelSuppThreshold_U8_U8_7x7_L2NORM(AgoNode * node, AgoKernelCommand cmd)
{
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
        // TBD: not implemented yet
    }
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_CannySuppThreshold_U8(node, VX_DF_IMAGE_U8, 3, 3);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_CannySobelSuppThreshold_U8XY_U8_3x3_L1NORM(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * oStack = node->paramList[1];
		AgoData * iImg = node->paramList[2];
		AgoData * iThr = node->paramList[3];
		oStack->u.cannystack.stackTop = 0;
		if (HafCpu_CannySobelSuppThreshold_U8XY_U8_3x3_L1NORM(oStack->u.cannystack.count, (ago_coord2d_ushort_t *)oStack->buffer, &oStack->u.cannystack.stackTop,
															  oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, 
															  iImg->buffer, iImg->u.img.stride_in_bytes,
															  iThr->u.thr.threshold_lower, iThr->u.thr.threshold_upper, node->localDataPtr))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_CannySuppThreshold_U8XY(node, VX_DF_IMAGE_U8, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		int alignedStride = (node->paramList[0]->u.img.stride_in_bytes + 15) & ~15;
		node->localDataSize = ((2 * alignedStride * node->paramList[0]->u.img.height) + (6 * alignedStride)) * sizeof(vx_int16);
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[2];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_CannySobelSuppThreshold_U8XY_U8_3x3_L2NORM(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * oStack = node->paramList[1];
		AgoData * iImg = node->paramList[2];
		AgoData * iThr = node->paramList[3];
		oStack->u.cannystack.stackTop = 0;
		if (HafCpu_CannySobelSuppThreshold_U8XY_U8_3x3_L2NORM(oStack->u.cannystack.count, (ago_coord2d_ushort_t *)oStack->buffer, &oStack->u.cannystack.stackTop,
															  oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, 
															  iImg->buffer, iImg->u.img.stride_in_bytes,
															  iThr->u.thr.threshold_lower, iThr->u.thr.threshold_upper))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_CannySuppThreshold_U8XY(node, VX_DF_IMAGE_U8, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[2];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_CannySobelSuppThreshold_U8XY_U8_5x5_L1NORM(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * oStack = node->paramList[1];
		AgoData * iImg = node->paramList[2];
		AgoData * iThr = node->paramList[3];
		oStack->u.cannystack.stackTop = 0;
		if (HafCpu_CannySobelSuppThreshold_U8XY_U8_5x5_L1NORM(oStack->u.cannystack.count, (ago_coord2d_ushort_t *)oStack->buffer, &oStack->u.cannystack.stackTop,
															  oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, 
															  iImg->buffer, iImg->u.img.stride_in_bytes,
															  iThr->u.thr.threshold_lower, iThr->u.thr.threshold_upper))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_CannySuppThreshold_U8XY(node, VX_DF_IMAGE_U8, 2, 2);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[2];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 2, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 2, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 2, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 2, 0);
	}
	return status;
}

int agoKernel_CannySobelSuppThreshold_U8XY_U8_5x5_L2NORM(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * oStack = node->paramList[1];
		AgoData * iImg = node->paramList[2];
		AgoData * iThr = node->paramList[3];
		oStack->u.cannystack.stackTop = 0;
		if (HafCpu_CannySobelSuppThreshold_U8XY_U8_5x5_L2NORM(oStack->u.cannystack.count, (ago_coord2d_ushort_t *)oStack->buffer, &oStack->u.cannystack.stackTop,
															  oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, 
															  iImg->buffer, iImg->u.img.stride_in_bytes,
															  iThr->u.thr.threshold_lower, iThr->u.thr.threshold_upper))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_CannySuppThreshold_U8XY(node, VX_DF_IMAGE_U8, 2, 2);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[2];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 2, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 2, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 2, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 2, 0);
	}
	return status;
}

int agoKernel_CannySobelSuppThreshold_U8XY_U8_7x7_L1NORM(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * oStack = node->paramList[1];
		AgoData * iImg = node->paramList[2];
		AgoData * iThr = node->paramList[3];
		oStack->u.cannystack.stackTop = 0;
		if (HafCpu_CannySobelSuppThreshold_U8XY_U8_7x7_L1NORM(oStack->u.cannystack.count, (ago_coord2d_ushort_t *)oStack->buffer, &oStack->u.cannystack.stackTop,
															  oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, 
															  iImg->buffer, iImg->u.img.stride_in_bytes,
															  iThr->u.thr.threshold_lower, iThr->u.thr.threshold_upper))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_CannySuppThreshold_U8XY(node, VX_DF_IMAGE_U8, 3, 3);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[2];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 3, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 3, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 3, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 3, 0);
	}
	return status;
}

int agoKernel_CannySobelSuppThreshold_U8XY_U8_7x7_L2NORM(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * oStack = node->paramList[1];
		AgoData * iImg = node->paramList[2];
		AgoData * iThr = node->paramList[3];
		oStack->u.cannystack.stackTop = 0;
		if (HafCpu_CannySobelSuppThreshold_U8XY_U8_7x7_L2NORM(oStack->u.cannystack.count, (ago_coord2d_ushort_t *)oStack->buffer, &oStack->u.cannystack.stackTop,
															  oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes, 
															  iImg->buffer, iImg->u.img.stride_in_bytes,
															  iThr->u.thr.threshold_lower, iThr->u.thr.threshold_upper))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_CannySuppThreshold_U8XY(node, VX_DF_IMAGE_U8, 3, 3);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[2];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 3, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 3, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 3, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 3, 0);
	}
	return status;
}

int agoKernel_CannySobel_U16_U8_3x3_L1NORM(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_CannySobel_U16_U8_3x3_L1NORM(oImg->u.img.width, oImg->u.img.height, (vx_uint16 *)oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes, node->localDataPtr)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U16, VX_DF_IMAGE_U8, true, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		int alignedWidth = (oImg->u.img.width + 15) & ~15;
		node->localDataSize = (alignedWidth * 4) + 128;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_CannySobelFilters(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_CannySobel_U16_U8_3x3_L2NORM(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_CannySobel_U16_U8_3x3_L2NORM(oImg->u.img.width, oImg->u.img.height, (vx_uint16 *)oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes, node->localDataPtr)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U16, VX_DF_IMAGE_U8, true, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		int alignedWidth = (oImg->u.img.width + 15) & ~15;
		node->localDataSize = (alignedWidth * 4) + 128;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_CannySobelFilters(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_CannySobel_U16_U8_5x5_L1NORM(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_CannySobel_U16_U8_5x5_L1NORM(oImg->u.img.width, oImg->u.img.height, (vx_uint16 *)oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes, node->localDataPtr)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U16, VX_DF_IMAGE_U8, true, 2, 2);
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		int alignedWidth = (oImg->u.img.width + 15) & ~15;
		node->localDataSize = (alignedWidth * 4) + 128;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_CannySobelFilters(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 2, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 2, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 2, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 2, 0);
	}
	return status;
}

int agoKernel_CannySobel_U16_U8_5x5_L2NORM(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_CannySobel_U16_U8_5x5_L2NORM(oImg->u.img.width, oImg->u.img.height, (vx_uint16 *)oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes, node->localDataPtr)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U16, VX_DF_IMAGE_U8, true, 2, 2);
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		int alignedWidth = (oImg->u.img.width + 15) & ~15;
		node->localDataSize = (alignedWidth * 4) + 128;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_CannySobelFilters(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 2, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 2, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 2, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 2, 0);
	}
	return status;
}

int agoKernel_CannySobel_U16_U8_7x7_L1NORM(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_CannySobel_U16_U8_7x7_L1NORM(oImg->u.img.width, oImg->u.img.height, (vx_uint16 *)oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes, node->localDataPtr)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U16, VX_DF_IMAGE_U8, true, 3, 3);
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		int alignedWidth = (oImg->u.img.width + 15) & ~15;
		node->localDataSize = (alignedWidth * 4) + 128;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_CannySobelFilters(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 3, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 3, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 3, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 3, 0);
	}
	return status;
}

int agoKernel_CannySobel_U16_U8_7x7_L2NORM(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_CannySobel_U16_U8_7x7_L2NORM(oImg->u.img.width, oImg->u.img.height, (vx_uint16 *)oImg->buffer, oImg->u.img.stride_in_bytes, iImg->buffer, iImg->u.img.stride_in_bytes, node->localDataPtr)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U16, VX_DF_IMAGE_U8, true, 3, 3);
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		int alignedWidth = (oImg->u.img.width + 15) & ~15;
		node->localDataSize = (alignedWidth * 4) + 128;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_CannySobelFilters(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 3, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 3, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 3, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 3, 0);
	}
	return status;
}

int agoKernel_CannySuppThreshold_U8_U16_3x3(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		// TBD: not implemented yet
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_CannySuppThreshold_U8(node, VX_DF_IMAGE_U16, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_CannySuppThreshold(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
#if ENABLE_OPENCL        
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_CannySuppThreshold_U8XY_U16_3x3(AgoNode * node, AgoKernelCommand cmd)
{
    vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
    if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * oStack = node->paramList[1];
		AgoData * iImg = node->paramList[2];
		AgoData * iThr = node->paramList[3];
		oStack->u.cannystack.stackTop = 0;
		if (HafCpu_CannySuppThreshold_U8XY_U16_3x3(oStack->u.cannystack.count, (ago_coord2d_ushort_t *)oStack->buffer, &oStack->u.cannystack.stackTop,
			oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
			(vx_uint16 *)iImg->buffer, iImg->u.img.stride_in_bytes,
			iThr->u.thr.threshold_lower, iThr->u.thr.threshold_upper))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_CannySuppThreshold_U8XY(node, VX_DF_IMAGE_U16, 1, 1);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_CannySuppThreshold(node);
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[2];
		vx_uint32 width = out->u.img.width;
		vx_uint32 height = out->u.img.height;
		out->u.img.rect_valid.start_x = min(inp->u.img.rect_valid.start_x + 1, width);
		out->u.img.rect_valid.start_y = min(inp->u.img.rect_valid.start_y + 1, height);
		out->u.img.rect_valid.end_x = max((int)inp->u.img.rect_valid.end_x - 1, 0);
		out->u.img.rect_valid.end_y = max((int)inp->u.img.rect_valid.end_y - 1, 0);
	}
	return status;
}

int agoKernel_NonMaxSupp_XY_ANY_3x3(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oList = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		vx_uint32 numitems = 0;
		if (HafCpu_NonMaxSupp_XY_ANY_3x3((vx_uint32)oList->u.arr.capacity, (ago_keypoint_xys_t *)oList->buffer, &numitems,
			iImg->u.img.width, iImg->u.img.height, (vx_float32 *)iImg->buffer, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
		else
		{
			oList->u.arr.numitems = numitems;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_F32_AMD)
			return VX_ERROR_INVALID_FORMAT;
		else if (!node->paramList[1]->u.img.width || !node->paramList[1]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.arr.itemtype = AGO_TYPE_KEYPOINT_XYS;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = HafGpu_NonMaxSupp_XY_ANY_3x3(node);
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
			| AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL			
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL
#endif			
			;
		status = VX_SUCCESS;
	}
	return status;
}

int agoKernel_Remap_U8_U8_Nearest(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iMap = node->paramList[2];
		if (HafCpu_Remap_U8_U8_Nearest(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
									   iImg->u.img.width, iImg->u.img.height, iImg->buffer, iImg->u.img.stride_in_bytes,
									   (ago_coord2d_ushort_t *)iMap->buffer, iMap->u.remap.dst_width * sizeof(ago_coord2d_ushort_t)))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
		if (!status) {
			if (node->paramList[1]->u.img.width != node->paramList[2]->u.remap.src_width ||
				node->paramList[1]->u.img.height != node->paramList[2]->u.remap.src_height)
				return VX_ERROR_INVALID_DIMENSION;
			// set output image sizes are same as input image size
			vx_meta_format meta;
			meta = &node->metaList[0];
			meta->data.u.img.width = node->paramList[2]->u.remap.dst_width;
			meta->data.u.img.height = node->paramList[2]->u.remap.dst_height;
			meta->data.u.img.format = VX_DF_IMAGE_U8;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		char textBuffer[4096];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s(U8x8 * r, uint x, uint y, __global uchar * p, uint stride, uint width, uint height, __global uchar * remap_, uint remap_stride_in_bytes)\n"
			"{\n"
			"  __global int * remap = (__global int *) (remap_ + y * remap_stride_in_bytes + (x << 2));\n"
			"  U8x8 rv;\n"
			"  int map; uint v;\n"
			"#define COMPUTE x = ((map & 0xffff) + 4) >> 3; y = (map + 0x00040000) >> 19; v = p[mad24(stride, y, x)]\n"
			"  map = remap[0]; COMPUTE ; rv.s0  = v;\n"
			"  map = remap[1]; COMPUTE ; rv.s0 |= v << 8;\n"
			"  map = remap[2]; COMPUTE ; rv.s0 |= v << 16;\n"
			"  map = remap[3]; COMPUTE ; rv.s0 |= v << 24;\n"
			"  map = remap[4]; COMPUTE ; rv.s1  = v;\n"
			"  map = remap[5]; COMPUTE ; rv.s1 |= v << 8;\n"
			"  map = remap[6]; COMPUTE ; rv.s1 |= v << 16;\n"
			"  map = remap[7]; COMPUTE ; rv.s1 |= v << 24;\n"
			"#undef COMPUTE\n"
			"  *r = rv;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code = textBuffer;
		node->opencl_type = NODE_OPENCL_TYPE_MEM2REG | NODE_OPENCL_TYPE_NEED_IMGSIZE;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_Remap_U8_U8_Nearest_Constant(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iMap = node->paramList[2];
		if (HafCpu_Remap_U8_U8_Nearest_Constant(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
			iImg->u.img.width, iImg->u.img.height, iImg->buffer, iImg->u.img.stride_in_bytes,
			(ago_coord2d_ushort_t *)iMap->buffer, iMap->u.remap.dst_width * sizeof(ago_coord2d_ushort_t), node->paramList[3]->u.scalar.u.u))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
		if (!status) {
			if (node->paramList[1]->u.img.width != node->paramList[2]->u.remap.src_width ||
				node->paramList[1]->u.img.height != node->paramList[2]->u.remap.src_height)
				return VX_ERROR_INVALID_DIMENSION;
			if (node->paramList[3]->u.scalar.type != VX_TYPE_UINT8)
				return VX_ERROR_INVALID_FORMAT;
			// set output image sizes are same as input image size
			vx_meta_format meta;
			meta = &node->metaList[0];
			meta->data.u.img.width = node->paramList[2]->u.remap.dst_width;
			meta->data.u.img.height = node->paramList[2]->u.remap.dst_height;
			meta->data.u.img.format = VX_DF_IMAGE_U8;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		char textBuffer[4096];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s(U8x8 * r, uint x, uint y, __global uchar * p, uint stride, uint width, uint height, __global uchar * remap_, uint remap_stride_in_bytes, uint borderValue)\n"
			"{\n"
			"  __global int * remap = (__global int *) (remap_ + y * remap_stride_in_bytes + (x << 2));\n"
			"  U8x8 rv;\n"
			"  int map; uint mask, v;\n"
			"  width -= 1; height -= 1;\n"
			"#define COMPUTE x = ((map & 0xffff) + 4) >> 3; y = (map + 0x00040000) >> 19; mask = ((int)(x | (width - x) | y | (height - y))) >> 31; mask = ~mask; x &= mask; y &= mask; v = p[mad24(stride, y, x)]; v = bitselect(borderValue, v, mask)\n"
			"  map = remap[0]; COMPUTE ; rv.s0 = v;\n"
			"  map = remap[1]; COMPUTE ; rv.s0 |= v << 8;\n"
			"  map = remap[2]; COMPUTE ; rv.s0 |= v << 16;\n"
			"  map = remap[3]; COMPUTE ; rv.s0 |= v << 24;\n"
			"  map = remap[4]; COMPUTE ; rv.s1  = v;\n"
			"  map = remap[5]; COMPUTE ; rv.s1 |= v << 8;\n"
			"  map = remap[6]; COMPUTE ; rv.s1 |= v << 16;\n"
			"  map = remap[7]; COMPUTE ; rv.s1 |= v << 24;\n"
			"#undef COMPUTE\n"
			"  *r = rv;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
		node->opencl_type = NODE_OPENCL_TYPE_MEM2REG | NODE_OPENCL_TYPE_NEED_IMGSIZE;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_Remap_U8_U8_Bilinear(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iMap = node->paramList[2];
		if (HafCpu_Remap_U8_U8_Bilinear(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
			iImg->u.img.width, iImg->u.img.height, iImg->buffer, iImg->u.img.stride_in_bytes,
			(ago_coord2d_ushort_t *)iMap->buffer, iMap->u.remap.dst_width * sizeof(ago_coord2d_ushort_t)))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
		if (!status) {
			if (node->paramList[1]->u.img.width != node->paramList[2]->u.remap.src_width ||
				node->paramList[1]->u.img.height != node->paramList[2]->u.remap.src_height)
				return VX_ERROR_INVALID_DIMENSION;
			// set output image sizes are same as input image size
			vx_meta_format meta;
			meta = &node->metaList[0];
			meta->data.u.img.width = node->paramList[2]->u.remap.dst_width;
			meta->data.u.img.height = node->paramList[2]->u.remap.dst_height;
			meta->data.u.img.format = VX_DF_IMAGE_U8;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		agoCodeGenOpenCL_BilinearSample(node->opencl_code);
		agoCodeGenOpenCL_BilinearSampleFXY(node->opencl_code);
		char textBuffer[4096];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s(U8x8 * r, uint x, uint y, __global uchar * p, uint stride, uint width, uint height, __global uchar * remap_, uint remap_stride_in_bytes)\n"
			"{\n"
			"  __global int * remap = (__global int *) (remap_ + y * remap_stride_in_bytes + (x << 2));\n"
			"  U8x8 rv;\n"
			"  float4 f; int map;\n"
			"  map = remap[0]; f.s0 = BilinearSampleFXY(p, stride, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f);\n"
			"  map = remap[1]; f.s1 = BilinearSampleFXY(p, stride, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f);\n"
			"  map = remap[2]; f.s2 = BilinearSampleFXY(p, stride, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f);\n"
			"  map = remap[3]; f.s3 = BilinearSampleFXY(p, stride, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f);\n"
			"  rv.s0 = amd_pack(f);\n"
			"  map = remap[4]; f.s0 = BilinearSampleFXY(p, stride, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f);\n"
			"  map = remap[5]; f.s1 = BilinearSampleFXY(p, stride, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f);\n"
			"  map = remap[6]; f.s2 = BilinearSampleFXY(p, stride, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f);\n"
			"  map = remap[7]; f.s3 = BilinearSampleFXY(p, stride, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f);\n"
			"  rv.s1 = amd_pack(f);\n"
			"  *r = rv;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
		node->opencl_type = NODE_OPENCL_TYPE_MEM2REG | NODE_OPENCL_TYPE_NEED_IMGSIZE;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_Remap_U8_U8_Bilinear_Constant(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iMap = node->paramList[2];
		if (HafCpu_Remap_U8_U8_Bilinear_Constant(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
			iImg->u.img.width, iImg->u.img.height, iImg->buffer, iImg->u.img.stride_in_bytes,
			(ago_coord2d_ushort_t *)iMap->buffer, iMap->u.remap.dst_width * sizeof(ago_coord2d_ushort_t), node->paramList[3]->u.scalar.u.u))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
		if (!status) {
			if (node->paramList[1]->u.img.width != node->paramList[2]->u.remap.src_width ||
				node->paramList[1]->u.img.height != node->paramList[2]->u.remap.src_height)
				return VX_ERROR_INVALID_DIMENSION;
			if (node->paramList[3]->u.scalar.type != VX_TYPE_UINT8)
				return VX_ERROR_INVALID_FORMAT;
			// set output image sizes are same as input image size
			vx_meta_format meta;
			meta = &node->metaList[0];
			meta->data.u.img.width = node->paramList[2]->u.remap.dst_width;
			meta->data.u.img.height = node->paramList[2]->u.remap.dst_height;
			meta->data.u.img.format = VX_DF_IMAGE_U8;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		agoCodeGenOpenCL_BilinearSample(node->opencl_code);
		agoCodeGenOpenCL_BilinearSampleFXYConstantForRemap(node->opencl_code);
		char textBuffer[4096];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s(U8x8 * r, uint x, uint y, __global uchar * p, uint stride, uint width, uint height, __global uchar * remap_, uint remap_stride_in_bytes, uint borderValue)\n"
			"{\n"
			"  __global int * remap = (__global int *) (remap_ + y * remap_stride_in_bytes + (x << 2));\n"
			"  U8x8 rv;\n"
			"  float4 f; int map;\n"
			"  map = remap[0]; f.s0 = BilinearSampleFXYConstantForRemap(p, stride, width, height, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f, borderValue);\n"
			"  map = remap[1]; f.s1 = BilinearSampleFXYConstantForRemap(p, stride, width, height, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f, borderValue);\n"
			"  map = remap[2]; f.s2 = BilinearSampleFXYConstantForRemap(p, stride, width, height, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f, borderValue);\n"
			"  map = remap[3]; f.s3 = BilinearSampleFXYConstantForRemap(p, stride, width, height, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f, borderValue);\n"
			"  rv.s0 = amd_pack(f);\n"
			"  map = remap[4]; f.s0 = BilinearSampleFXYConstantForRemap(p, stride, width, height, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f, borderValue);\n"
			"  map = remap[5]; f.s1 = BilinearSampleFXYConstantForRemap(p, stride, width, height, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f, borderValue);\n"
			"  map = remap[6]; f.s2 = BilinearSampleFXYConstantForRemap(p, stride, width, height, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f, borderValue);\n"
			"  map = remap[7]; f.s3 = BilinearSampleFXYConstantForRemap(p, stride, width, height, ((map << 16) >> 16) * 0.125f, (map >> 16) * 0.125f, borderValue);\n"
			"  rv.s1 = amd_pack(f);\n"
			"  *r = rv;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
		node->opencl_type = NODE_OPENCL_TYPE_MEM2REG | NODE_OPENCL_TYPE_NEED_IMGSIZE;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_Remap_U24_U24_Bilinear(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		// not implemented yet
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_RGB, VX_DF_IMAGE_RGB);
		if (!status) {
			if (node->paramList[1]->u.img.width != node->paramList[2]->u.remap.src_width ||
				node->paramList[1]->u.img.height != node->paramList[2]->u.remap.src_height)
				return VX_ERROR_INVALID_DIMENSION;
			// set output image sizes are same as input image size
			vx_meta_format meta;
			meta = &node->metaList[0];
			meta->data.u.img.width = node->paramList[2]->u.remap.dst_width;
			meta->data.u.img.height = node->paramList[2]->u.remap.dst_height;
			meta->data.u.img.format = VX_DF_IMAGE_RGB;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		char textBuffer[1024];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s(U24x8 * r, uint x, uint y, __global uchar * p, uint stride, uint width, uint height, __global uchar * remap_, uint remap_stride_in_bytes)\n"
			"{\n"
			"  uint QF = %d;\n"
			), node->opencl_name, node->paramList[2]->u.remap.remap_fractional_bits);
		node->opencl_code += textBuffer;
		node->opencl_code += OPENCL_FORMAT(
			"  uint invalidPix = amd_pack((float4)(0.0f, 0.0f, 0.0f, 0.0f));\n"
			"  float mulfactor;\n"
			"  __global int * remap = (__global int *) (remap_ + y * remap_stride_in_bytes + (x << 2));\n"
			"  U24x8 rv;\n"
			"  float4 f; uint map, sx, sy, offset; uint3 px0, px1; __global uchar * pt; float4 mf;\n"
			"  uint QFB = (1 << QF) - 1; float QFM = 1.0f / (1 << QF);\n"
			"  // pixel[0]\n"
			"  map = remap[0]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + (sx >> QF) * 3; mulfactor = 1.0f;\n"
			"  if(sx == 0xffff && sy == 0xffff) { sx = 0; sy = 0; mulfactor = 0.0f; }\n"
			"  pt = p + (offset & ~3); px0 = vload3(0, (__global uint *)pt); px1 = vload3(0, (__global uint *)(pt + stride)); px0.s0 = amd_bytealign(px0.s1, px0.s0, offset); px0.s1 = amd_bytealign(px0.s2, px0.s1, offset); px1.s0 = amd_bytealign(px1.s1, px1.s0, offset); px1.s1 = amd_bytealign(px1.s2, px1.s1, offset); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
			"  f.s0 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack3(px0.s0) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack3(px1.s0) * mf.s0) * mf.s1;\n"
			"  f.s1 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s2 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s012 *= mulfactor;\n"
			"  // pixel[1]\n"
			"  map = remap[1]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + (sx >> QF) * 3; mulfactor = 1.0f;\n"
			"  if(sx == 0xffff && sy == 0xffff) { sx = 0; sy = 0; mulfactor = 0.0f; }\n"
			"  pt = p + (offset & ~3); px0 = vload3(0, (__global uint *)pt); px1 = vload3(0, (__global uint *)(pt + stride)); px0.s0 = amd_bytealign(px0.s1, px0.s0, offset); px0.s1 = amd_bytealign(px0.s2, px0.s1, offset); px1.s0 = amd_bytealign(px1.s1, px1.s0, offset); px1.s1 = amd_bytealign(px1.s2, px1.s1, offset); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
			"  f.s3 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack3(px0.s0) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack3(px1.s0) * mf.s0) * mf.s1;\n"
			"  f.s3 *= mulfactor;\n"
			"  rv.s0 = amd_pack(f);\n"
			"  f.s0 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s1 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s01 *= mulfactor;\n"
			"  // pixel[2]\n"
			"  map = remap[2]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + (sx >> QF) * 3; mulfactor = 1.0f;\n"
			"  if(sx == 0xffff && sy == 0xffff) { sx = 0; sy = 0; mulfactor = 0.0f; }\n"
			"  pt = p + (offset & ~3); px0 = vload3(0, (__global uint *)pt); px1 = vload3(0, (__global uint *)(pt + stride)); px0.s0 = amd_bytealign(px0.s1, px0.s0, offset); px0.s1 = amd_bytealign(px0.s2, px0.s1, offset); px1.s0 = amd_bytealign(px1.s1, px1.s0, offset); px1.s1 = amd_bytealign(px1.s2, px1.s1, offset); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
			"  f.s2 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack3(px0.s0) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack3(px1.s0) * mf.s0) * mf.s1;\n"
			"  f.s3 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s23 *= mulfactor;\n"
			"  rv.s1 = amd_pack(f);\n"
			"  f.s0 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s0 *= mulfactor;\n"
			"  // pixel[3]\n"
			"  map = remap[3]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + (sx >> QF) * 3; mulfactor = 1.0f;\n"
			"  if(sx == 0xffff && sy == 0xffff) { sx = 0; sy = 0; mulfactor = 0.0f; }\n"
			"  pt = p + (offset & ~3); px0 = vload3(0, (__global uint *)pt); px1 = vload3(0, (__global uint *)(pt + stride)); px0.s0 = amd_bytealign(px0.s1, px0.s0, offset); px0.s1 = amd_bytealign(px0.s2, px0.s1, offset); px1.s0 = amd_bytealign(px1.s1, px1.s0, offset); px1.s1 = amd_bytealign(px1.s2, px1.s1, offset); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
			"  f.s1 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack3(px0.s0) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack3(px1.s0) * mf.s0) * mf.s1;\n"
			"  f.s2 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s3 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s123 *= mulfactor;\n"
			"  rv.s2 = amd_pack(f);\n"
			"  // pixel[4]\n"
			"  map = remap[4]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + (sx >> QF) * 3; mulfactor = 1.0f;\n"
			"  if(sx == 0xffff && sy == 0xffff) { sx = 0; sy = 0; mulfactor = 0.0f; }\n"
			"  pt = p + (offset & ~3); px0 = vload3(0, (__global uint *)pt); px1 = vload3(0, (__global uint *)(pt + stride)); px0.s0 = amd_bytealign(px0.s1, px0.s0, offset); px0.s1 = amd_bytealign(px0.s2, px0.s1, offset); px1.s0 = amd_bytealign(px1.s1, px1.s0, offset); px1.s1 = amd_bytealign(px1.s2, px1.s1, offset); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
			"  f.s0 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack3(px0.s0) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack3(px1.s0) * mf.s0) * mf.s1;\n"
			"  f.s1 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s2 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s012 *= mulfactor;\n"
			"  // pixel[5]\n"
			"  map = remap[5]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + (sx >> QF) * 3; mulfactor = 1.0f;\n"
			"  if(sx == 0xffff && sy == 0xffff) { sx = 0; sy = 0; mulfactor = 0.0f; }\n"
			"  pt = p + (offset & ~3); px0 = vload3(0, (__global uint *)pt); px1 = vload3(0, (__global uint *)(pt + stride)); px0.s0 = amd_bytealign(px0.s1, px0.s0, offset); px0.s1 = amd_bytealign(px0.s2, px0.s1, offset); px1.s0 = amd_bytealign(px1.s1, px1.s0, offset); px1.s1 = amd_bytealign(px1.s2, px1.s1, offset); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
			"  f.s3 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack3(px0.s0) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack3(px1.s0) * mf.s0) * mf.s1;\n"
			"  f.s3 *= mulfactor;\n"
			"  rv.s3 = amd_pack(f);\n"
			"  f.s0 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s1 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s01 *= mulfactor;\n"
			"  // pixel[6]\n"
			"  map = remap[6]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + (sx >> QF) * 3; mulfactor = 1.0f;\n"
			"  if(sx == 0xffff && sy == 0xffff) { sx = 0; sy = 0; mulfactor = 0.0f; }\n"
			"  pt = p + (offset & ~3); px0 = vload3(0, (__global uint *)pt); px1 = vload3(0, (__global uint *)(pt + stride)); px0.s0 = amd_bytealign(px0.s1, px0.s0, offset); px0.s1 = amd_bytealign(px0.s2, px0.s1, offset); px1.s0 = amd_bytealign(px1.s1, px1.s0, offset); px1.s1 = amd_bytealign(px1.s2, px1.s1, offset); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
			"  f.s2 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack3(px0.s0) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack3(px1.s0) * mf.s0) * mf.s1;\n"
			"  f.s3 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s23 *= mulfactor;\n"
			"  rv.s4 = amd_pack(f);\n"
			"  f.s0 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s0 *= mulfactor;\n"
			"  // pixel[7]\n"
			"  map = remap[7]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + (sx >> QF) * 3; mulfactor = 1.0f;\n"
			"  if(sx == 0xffff && sy == 0xffff) { sx = 0; sy = 0; mulfactor = 0.0f; }\n"
			"  pt = p + (offset & ~3); px0 = vload3(0, (__global uint *)pt); px1 = vload3(0, (__global uint *)(pt + stride)); px0.s0 = amd_bytealign(px0.s1, px0.s0, offset); px0.s1 = amd_bytealign(px0.s2, px0.s1, offset); px1.s0 = amd_bytealign(px1.s1, px1.s0, offset); px1.s1 = amd_bytealign(px1.s2, px1.s1, offset); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
			"  f.s1 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack3(px0.s0) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack3(px1.s0) * mf.s0) * mf.s1;\n"
			"  f.s2 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s3 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s123 *= mulfactor;\n"
			"  rv.s5 = amd_pack(f);\n"
			"  *r = rv;\n"
			"}\n"
			);
		node->opencl_type = NODE_OPENCL_TYPE_MEM2REG | NODE_OPENCL_TYPE_NEED_IMGSIZE;
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
#if ENABLE_OPENCL		
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif			
			;
		status = VX_SUCCESS;
	}
	return status;
}

int agoKernel_Remap_U24_U32_Bilinear(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		// not implemented yet
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_RGB, VX_DF_IMAGE_RGBX);
		if (!status) {
			if (node->paramList[1]->u.img.width != node->paramList[2]->u.remap.src_width ||
				node->paramList[1]->u.img.height != node->paramList[2]->u.remap.src_height)
				return VX_ERROR_INVALID_DIMENSION;
			// set output image sizes are same as input image size
			vx_meta_format meta;
			meta = &node->metaList[0];
			meta->data.u.img.width = node->paramList[2]->u.remap.dst_width;
			meta->data.u.img.height = node->paramList[2]->u.remap.dst_height;
			meta->data.u.img.format = VX_DF_IMAGE_RGB;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		char textBuffer[1024];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s(U24x8 * r, uint x, uint y, __global uchar * p, uint stride, uint width, uint height, __global uchar * remap_, uint remap_stride_in_bytes)\n"
			"{\n"
			"  uint QF = %d;\n"
			), node->opencl_name, node->paramList[2]->u.remap.remap_fractional_bits);
		node->opencl_code += textBuffer;
		node->opencl_code += OPENCL_FORMAT(
			"  uint invalidPix = amd_pack((float4)(0.0f, 0.0f, 0.0f, 0.0f));\n"
			"  float mulfactor;\n"
			"  __global int * remap = (__global int *) (remap_ + y * remap_stride_in_bytes + (x << 2));\n"
			"  U24x8 rv;\n"
			"  float4 f; uint map, sx, sy, offset; uint2 px0, px1; __global uchar * pt; float4 mf;\n"
			"  uint QFB = (1 << QF) - 1; float QFM = 1.0f / (1 << QF);\n"
			"  // pixel[0]\n"
			"  map = remap[0]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + ((sx >> QF) << 2); mulfactor = 1.0f;\n"
			"  if(sx == 0xffff && sy == 0xffff) { sx = 0; sy = 0; mulfactor = 0.0f; }\n"
			"  pt = p + offset; px0 = vload2(0, (__global uint *)pt); px1 = vload2(0, (__global uint *)(pt + stride)); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
			"  f.s0 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s1 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s2 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack2(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack2(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s012 *= mulfactor;\n"
			"  // pixel[1]\n"
			"  map = remap[1]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + ((sx >> QF) << 2); mulfactor = 1.0f;\n"
			"  if(sx == 0xffff && sy == 0xffff) { sx = 0; sy = 0; mulfactor = 0.0f; }\n"
			"  pt = p + offset; px0 = vload2(0, (__global uint *)pt); px1 = vload2(0, (__global uint *)(pt + stride)); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
			"  f.s3 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s3 *= mulfactor;\n"
			"  rv.s0 = amd_pack(f);\n"
			"  f.s0 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s1 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack2(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack2(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s01 *= mulfactor;\n"
			"  // pixel[2]\n"
			"  map = remap[2]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + ((sx >> QF) << 2); mulfactor = 1.0f;\n"
			"  if(sx == 0xffff && sy == 0xffff) { sx = 0; sy = 0; mulfactor = 0.0f; }\n"
			"  pt = p + offset; px0 = vload2(0, (__global uint *)pt); px1 = vload2(0, (__global uint *)(pt + stride)); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
			"  f.s2 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s3 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s23 *= mulfactor;\n"
			"  rv.s1 = amd_pack(f);\n"
			"  f.s0 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack2(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack2(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s0 *= mulfactor;\n"
			"  // pixel[3]\n"
			"  map = remap[3]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + ((sx >> QF) << 2); mulfactor = 1.0f;\n"
			"  if(sx == 0xffff && sy == 0xffff) { sx = 0; sy = 0; mulfactor = 0.0f; }\n"
			"  pt = p + offset; px0 = vload2(0, (__global uint *)pt); px1 = vload2(0, (__global uint *)(pt + stride)); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
			"  f.s1 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s2 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s3 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack2(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack2(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s123 *= mulfactor;\n"
			"  rv.s2 = amd_pack(f);\n"
			"  // pixel[4]\n"
			"  map = remap[4]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + ((sx >> QF) << 2); mulfactor = 1.0f;\n"
			"  if(sx == 0xffff && sy == 0xffff) { sx = 0; sy = 0; mulfactor = 0.0f; }\n"
			"  pt = p + offset; px0 = vload2(0, (__global uint *)pt); px1 = vload2(0, (__global uint *)(pt + stride)); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
			"  f.s0 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s1 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s2 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack2(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack2(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s012 *= mulfactor;\n"
			"  // pixel[5]\n"
			"  map = remap[5]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + ((sx >> QF) << 2); mulfactor = 1.0f;\n"
			"  if(sx == 0xffff && sy == 0xffff) { sx = 0; sy = 0; mulfactor = 0.0f; }\n"
			"  pt = p + offset; px0 = vload2(0, (__global uint *)pt); px1 = vload2(0, (__global uint *)(pt + stride)); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
			"  f.s3 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s3 *= mulfactor;\n"
			"  rv.s3 = amd_pack(f);\n"
			"  f.s0 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s1 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack2(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack2(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s01 *= mulfactor;\n"
			"  // pixel[6]\n"
			"  map = remap[6]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + ((sx >> QF) << 2); mulfactor = 1.0f;\n"
			"  if(sx == 0xffff && sy == 0xffff) { sx = 0; sy = 0; mulfactor = 0.0f; }\n"
			"  pt = p + offset; px0 = vload2(0, (__global uint *)pt); px1 = vload2(0, (__global uint *)(pt + stride)); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
			"  f.s2 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s3 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s23 *= mulfactor;\n"
			"  rv.s4 = amd_pack(f);\n"
			"  f.s0 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack2(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack2(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s0 *= mulfactor;\n"
			"  // pixel[7]\n"
			"  map = remap[7]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + ((sx >> QF) << 2); mulfactor = 1.0f;\n"
			"  if(sx == 0xffff && sy == 0xffff) { sx = 0; sy = 0; mulfactor = 0.0f; }\n"
			"  pt = p + offset; px0 = vload2(0, (__global uint *)pt); px1 = vload2(0, (__global uint *)(pt + stride)); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
			"  f.s1 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s2 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s3 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack2(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack2(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s123 *= mulfactor;\n"
			"  rv.s5 = amd_pack(f);\n"
			"  *r = rv;\n"
			"}\n"
			);
		node->opencl_type = NODE_OPENCL_TYPE_MEM2REG | NODE_OPENCL_TYPE_NEED_IMGSIZE;
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
#if ENABLE_OPENCL		
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif			
			;
		status = VX_SUCCESS;
	}
	return status;
}

int agoKernel_Remap_U32_U32_Bilinear(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		// not implemented yet
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_RGBX, VX_DF_IMAGE_RGBX);
		if (!status) {
			if (node->paramList[1]->u.img.width != node->paramList[2]->u.remap.src_width ||
				node->paramList[1]->u.img.height != node->paramList[2]->u.remap.src_height)
				return VX_ERROR_INVALID_DIMENSION;
			// set output image sizes are same as input image size
			vx_meta_format meta;
			meta = &node->metaList[0];
			meta->data.u.img.width = node->paramList[2]->u.remap.dst_width;
			meta->data.u.img.height = node->paramList[2]->u.remap.dst_height;
			meta->data.u.img.format = VX_DF_IMAGE_RGBX;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		char textBuffer[1024];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s(U32x8 * r, uint x, uint y, __global uchar * p, uint stride, uint width, uint height, __global uchar * remap_, uint remap_stride_in_bytes)\n"
			"{\n"
			"  uint QF = %d;\n"
			), node->opencl_name, node->paramList[2]->u.remap.remap_fractional_bits);
		node->opencl_code += textBuffer;
		node->opencl_code += OPENCL_FORMAT(
			"  uint invalidPix = amd_pack((float4)(0.0f));\n"
			"  bool isSrcInvalid;\n"
			"  __global int * remap = (__global int *) (remap_ + y * remap_stride_in_bytes + (x << 2));\n"
			"  U32x8 rv;\n"
			"  float4 f; uint map, sx, sy, offset; uint2 px0, px1; __global uchar * pt; float4 mf;\n"
			"  uint QFB = (1 << QF) - 1; float QFM = 1.0f / (1 << QF);\n"
			"  // pixel[0]\n"
			"  map = remap[0]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + ((sx >> QF) << 2); isSrcInvalid = false;\n"
			"  if(sx == 0xffff && sy == 0xffff) { isSrcInvalid = true; sx = 1 << QF; sy = 1 << QF; }\n"
			"  pt = p + offset; px0 = vload2(0, (__global uint *)pt); px1 = vload2(0, (__global uint *)(pt + stride)); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1; \n"
			"  f.s0 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s1 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s2 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack2(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack2(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s3 = (amd_unpack3(px0.s0) * mf.s2 + amd_unpack3(px0.s1) * mf.s0) * mf.s3 + (amd_unpack3(px1.s0) * mf.s2 + amd_unpack3(px1.s1) * mf.s0) * mf.s1;\n"
			"  rv.s0 = select(amd_pack(f), invalidPix, isSrcInvalid);\n"
			"  // pixel[1]\n"
			"  map = remap[1]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + ((sx >> QF) << 2); isSrcInvalid = false;\n"
			"  if(sx == 0xffff && sy == 0xffff) { isSrcInvalid = true; sx = 1 << QF; sy = 1 << QF; }\n"
			"   pt = p + offset; px0 = vload2(0, (__global uint *)pt); px1 = vload2(0, (__global uint *)(pt + stride)); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
			"  f.s0 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s1 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s2 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack2(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack2(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s3 = (amd_unpack3(px0.s0) * mf.s2 + amd_unpack3(px0.s1) * mf.s0) * mf.s3 + (amd_unpack3(px1.s0) * mf.s2 + amd_unpack3(px1.s1) * mf.s0) * mf.s1;\n"
			"  rv.s1 = select(amd_pack(f), invalidPix, isSrcInvalid);\n"
			"  // pixel[2]\n"
			"  map = remap[2]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + ((sx >> QF) << 2); isSrcInvalid = false;\n"
			"  if(sx == 0xffff && sy == 0xffff) { isSrcInvalid = true; sx = 1 << QF; sy = 1 << QF; }\n"
			"   pt = p + offset; px0 = vload2(0, (__global uint *)pt); px1 = vload2(0, (__global uint *)(pt + stride)); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
			"  f.s0 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s1 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s2 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack2(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack2(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s3 = (amd_unpack3(px0.s0) * mf.s2 + amd_unpack3(px0.s1) * mf.s0) * mf.s3 + (amd_unpack3(px1.s0) * mf.s2 + amd_unpack3(px1.s1) * mf.s0) * mf.s1;\n"
			"  rv.s2 = select(amd_pack(f), invalidPix, isSrcInvalid);\n"
			"  // pixel[3]\n"
			"  map = remap[3]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + ((sx >> QF) << 2); isSrcInvalid = false;\n"
			"  if(sx == 0xffff && sy == 0xffff) { isSrcInvalid = true; sx = 1 << QF; sy = 1 << QF; }\n"
			"   pt = p + offset; px0 = vload2(0, (__global uint *)pt); px1 = vload2(0, (__global uint *)(pt + stride)); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
			"  f.s0 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s1 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s2 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack2(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack2(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s3 = (amd_unpack3(px0.s0) * mf.s2 + amd_unpack3(px0.s1) * mf.s0) * mf.s3 + (amd_unpack3(px1.s0) * mf.s2 + amd_unpack3(px1.s1) * mf.s0) * mf.s1;\n"
			"  rv.s3 = select(amd_pack(f), invalidPix, isSrcInvalid);\n"
			"  // pixel[4]\n"
			"  map = remap[4]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + ((sx >> QF) << 2); isSrcInvalid = false;\n"
			"  if(sx == 0xffff && sy == 0xffff) { isSrcInvalid = true; sx = 1 << QF; sy = 1 << QF; }\n"
			"   pt = p + offset; px0 = vload2(0, (__global uint *)pt); px1 = vload2(0, (__global uint *)(pt + stride)); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
			"  f.s0 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s1 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s2 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack2(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack2(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s3 = (amd_unpack3(px0.s0) * mf.s2 + amd_unpack3(px0.s1) * mf.s0) * mf.s3 + (amd_unpack3(px1.s0) * mf.s2 + amd_unpack3(px1.s1) * mf.s0) * mf.s1;\n"
			"  rv.s4 = select(amd_pack(f), invalidPix, isSrcInvalid);\n"
			"  // pixel[5]\n"
			"  map = remap[5]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + ((sx >> QF) << 2); isSrcInvalid = false;\n"
			"  if(sx == 0xffff && sy == 0xffff) { isSrcInvalid = true; sx = 1 << QF; sy = 1 << QF; }\n"
			"   pt = p + offset; px0 = vload2(0, (__global uint *)pt); px1 = vload2(0, (__global uint *)(pt + stride)); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
			"  f.s0 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s1 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s2 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack2(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack2(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s3 = (amd_unpack3(px0.s0) * mf.s2 + amd_unpack3(px0.s1) * mf.s0) * mf.s3 + (amd_unpack3(px1.s0) * mf.s2 + amd_unpack3(px1.s1) * mf.s0) * mf.s1;\n"
			"  rv.s5 = select(amd_pack(f), invalidPix, isSrcInvalid);\n"
			"  // pixel[6]\n"
			"  map = remap[6]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + ((sx >> QF) << 2); isSrcInvalid = false;\n"
			"  if(sx == 0xffff && sy == 0xffff) { isSrcInvalid = true; sx = 1 << QF; sy = 1 << QF; }\n"
			"   pt = p + offset; px0 = vload2(0, (__global uint *)pt); px1 = vload2(0, (__global uint *)(pt + stride)); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
			"  f.s0 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s1 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s2 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack2(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack2(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s3 = (amd_unpack3(px0.s0) * mf.s2 + amd_unpack3(px0.s1) * mf.s0) * mf.s3 + (amd_unpack3(px1.s0) * mf.s2 + amd_unpack3(px1.s1) * mf.s0) * mf.s1;\n"
			"  rv.s6 = select(amd_pack(f), invalidPix, isSrcInvalid);\n"
			"  // pixel[7]\n"
			"  map = remap[7]; sx = map & 0xffff; sy = (map >> 16); offset = (sy >> QF) * stride + ((sx >> QF) << 2); isSrcInvalid = false;\n"
			"  if(sx == 0xffff && sy == 0xffff) { isSrcInvalid = true; sx = 1 << QF; sy = 1 << QF; }\n"
			"   pt = p + offset; px0 = vload2(0, (__global uint *)pt); px1 = vload2(0, (__global uint *)(pt + stride)); mf.s0 = (sx & QFB) * QFM; mf.s1 = (sy & QFB) * QFM; mf.s2 = 1.0f - mf.s0; mf.s3 = 1.0f - mf.s1;\n"
			"  f.s0 = (amd_unpack0(px0.s0) * mf.s2 + amd_unpack0(px0.s1) * mf.s0) * mf.s3 + (amd_unpack0(px1.s0) * mf.s2 + amd_unpack0(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s1 = (amd_unpack1(px0.s0) * mf.s2 + amd_unpack1(px0.s1) * mf.s0) * mf.s3 + (amd_unpack1(px1.s0) * mf.s2 + amd_unpack1(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s2 = (amd_unpack2(px0.s0) * mf.s2 + amd_unpack2(px0.s1) * mf.s0) * mf.s3 + (amd_unpack2(px1.s0) * mf.s2 + amd_unpack2(px1.s1) * mf.s0) * mf.s1;\n"
			"  f.s3 = (amd_unpack3(px0.s0) * mf.s2 + amd_unpack3(px0.s1) * mf.s0) * mf.s3 + (amd_unpack3(px1.s0) * mf.s2 + amd_unpack3(px1.s1) * mf.s0) * mf.s1;\n"
			"  rv.s7 = select(amd_pack(f), invalidPix, isSrcInvalid);\n"
			"  *r = rv;\n"
			"}\n"
			);
		node->opencl_type = NODE_OPENCL_TYPE_MEM2REG | NODE_OPENCL_TYPE_NEED_IMGSIZE;
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
#if ENABLE_OPENCL		
			| AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif			
			;
		status = VX_SUCCESS;
	}
	return status;
}

int agoKernel_WarpAffine_U8_U8_Nearest(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iMat = node->paramList[2];
		if (HafCpu_WarpAffine_U8_U8_Nearest(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
			iImg->u.img.width, iImg->u.img.height, iImg->buffer, iImg->u.img.stride_in_bytes, (ago_affine_matrix_t *)iMat->buffer, node->localDataPtr))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
		if (!status) {
			if (node->paramList[2]->u.mat.type != VX_TYPE_FLOAT32)
				return VX_ERROR_INVALID_TYPE;
			if (node->paramList[2]->u.mat.columns != 2 || node->paramList[2]->u.mat.rows != 3)
				return VX_ERROR_INVALID_DIMENSION;
			// output image dimensions have no constraints
			vx_meta_format meta;
			meta = &node->metaList[0];
			meta->data.u.img.width = node->paramList[0]->u.img.width;
			meta->data.u.img.height = node->paramList[0]->u.img.height;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		int alignedWidth = (node->paramList[0]->u.img.width + 15) & ~15;		// Next highest multiple of 16, so that the buffer is aligned for all three lines
		node->localDataSize = 2 * alignedWidth*sizeof(float);				// 2 rows (+some extra) worth of scratch memory			
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		char textBuffer[4096];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s(U8x8 * r, uint x, uint y, __global uchar * p, uint stride, uint width, uint height, ago_affine_matrix_t matrix)\n"
			"{\n"
			"  U8x8 rv;\n"
			"  float sx, sy;\n"
			"  float dx = (float)x, dy = (float)y;\n"
			"  sx = mad(dy, matrix.M[1][0], matrix.M[2][0]); sx = mad(dx, matrix.M[0][0], sx);\n"
			"  sy = mad(dy, matrix.M[1][1], matrix.M[2][1]); sy = mad(dx, matrix.M[0][1], sy);\n"
			"  rv.s0 = p[mad24(stride, (uint)sy, (uint)sx)];\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; rv.s0 |= p[mad24(stride, (uint)sy, (uint)sx)] << 8;\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; rv.s0 |= p[mad24(stride, (uint)sy, (uint)sx)] << 16;\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; rv.s0 |= p[mad24(stride, (uint)sy, (uint)sx)] << 24;\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; rv.s1  = p[mad24(stride, (uint)sy, (uint)sx)];\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; rv.s1 |= p[mad24(stride, (uint)sy, (uint)sx)] << 8;\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; rv.s1 |= p[mad24(stride, (uint)sy, (uint)sx)] << 16;\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; rv.s1 |= p[mad24(stride, (uint)sy, (uint)sx)] << 24;\n"
			"  *r = rv;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
		node->opencl_type = NODE_OPENCL_TYPE_MEM2REG | NODE_OPENCL_TYPE_NEED_IMGSIZE;
		node->opencl_param_as_value_mask |= (1 << 2); // matrix parameter needs to be passed by value
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_WarpAffine_U8_U8_Nearest_Constant(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iMat = node->paramList[2];
		if (HafCpu_WarpAffine_U8_U8_Nearest_Constant(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
			iImg->u.img.width, iImg->u.img.height, iImg->buffer, iImg->u.img.stride_in_bytes, (ago_affine_matrix_t *)iMat->buffer, node->paramList[3]->u.scalar.u.u, node->localDataPtr))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
		if (!status) {
			if (node->paramList[2]->u.mat.type != VX_TYPE_FLOAT32)
				return VX_ERROR_INVALID_TYPE;
			if (node->paramList[2]->u.mat.columns != 2 || node->paramList[2]->u.mat.rows != 3)
				return VX_ERROR_INVALID_DIMENSION;
			if (node->paramList[3]->u.scalar.type != VX_TYPE_UINT8)
				return VX_ERROR_INVALID_FORMAT;
			// output image dimensions have no constraints
			vx_meta_format meta;
			meta = &node->metaList[0];
			meta->data.u.img.width = node->paramList[0]->u.img.width;
			meta->data.u.img.height = node->paramList[0]->u.img.height;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		int alignedWidth = (node->paramList[0]->u.img.width + 15) & ~15;		// Next highest multiple of 16, so that the buffer is aligned for all three lines
		node->localDataSize = 2 * alignedWidth*sizeof(float);				// Three rows (+some extra) worth of scratch memory			
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		char textBuffer[4096];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s(U8x8 * r, uint x, uint y, __global uchar * p, uint stride, uint width, uint height, ago_affine_matrix_t matrix, uint border)\n"
			"{\n"
			"  U8x8 rv;\n"
			"  float sx, sy; uint mask, v;\n"
			"  float dx = (float)x, dy = (float)y;\n"
			"  sx = mad(dy, matrix.M[1][0], matrix.M[2][0]); sx = mad(dx, matrix.M[0][0], sx);\n"
			"  sy = mad(dy, matrix.M[1][1], matrix.M[2][1]); sy = mad(dx, matrix.M[0][1], sy);\n"
			"  x = (uint)(int)sx; y = (uint)(int)sy;\n"
			"  width -= 2; height -= 2;\n"
			"  mask = ((int)(x | (width - x) | y | (height - y))) >> 31; mask = ~mask;\n"
			"  x &= mask; y &= mask; v = p[mad24(stride, y, x)]; v = bitselect(border, v, mask); rv.s0 = v;\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; x = (uint)(int)sx; y = (uint)(int)sy; \n"
			"  mask = ((int)(x | (width - x) | y | (height - y))) >> 31; mask = ~mask;\n"
			"  x &= mask; y &= mask; v = p[mad24(stride, y, x)]; v = bitselect(border, v, mask); rv.s0 |= v << 8;\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; x = (uint)(int)sx; y = (uint)(int)sy;\n"
			"  mask = ((int)(x | (width - x) | y | (height - y))) >> 31; mask = ~mask;\n"
			"  x &= mask; y &= mask; v = p[mad24(stride, y, x)]; v = bitselect(border, v, mask); rv.s0 |= v << 16;\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; x = (uint)(int)sx; y = (uint)(int)sy;\n"
			"  mask = ((int)(x | (width - x) | y | (height - y))) >> 31; mask = ~mask;\n"
			"  x &= mask; y &= mask; v = p[mad24(stride, y, x)]; v = bitselect(border, v, mask); rv.s0 |= v << 24;\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; x = (uint)(int)sx; y = (uint)(int)sy;\n"
			"  mask = ((int)(x | (width - x) | y | (height - y))) >> 31; mask = ~mask;\n"
			"  x &= mask; y &= mask; v = p[mad24(stride, y, x)]; v = bitselect(border, v, mask); rv.s1 = v;\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; x = (uint)(int)sx; y = (uint)(int)sy;\n"
			"  mask = ((int)(x | (width - x) | y | (height - y))) >> 31; mask = ~mask;\n"
			"  x &= mask; y &= mask; v = p[mad24(stride, y, x)]; v = bitselect(border, v, mask); rv.s1 |= v << 8;\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; x = (uint)(int)sx; y = (uint)(int)sy;\n"
			"  mask = ((int)(x | (width - x) | y | (height - y))) >> 31; mask = ~mask;\n"
			"  x &= mask; y &= mask; v = p[mad24(stride, y, x)]; v = bitselect(border, v, mask); rv.s1 |= v << 16;\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; x = (uint)(int)sx; y = (uint)(int)sy;\n"
			"  mask = ((int)(x | (width - x) | y | (height - y))) >> 31; mask = ~mask;\n"
			"  x &= mask; y &= mask; v = p[mad24(stride, y, x)]; v = bitselect(border, v, mask); rv.s1 |= v << 24;\n"
			"  *r = rv;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
		node->opencl_type = NODE_OPENCL_TYPE_MEM2REG | NODE_OPENCL_TYPE_NEED_IMGSIZE;
		node->opencl_param_as_value_mask |= (1 << 2); // matrix parameter needs to be passed by value
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_WarpAffine_U8_U8_Bilinear(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iMat = node->paramList[2];
		if (HafCpu_WarpAffine_U8_U8_Bilinear(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
			iImg->u.img.width, iImg->u.img.height, iImg->buffer, iImg->u.img.stride_in_bytes, (ago_affine_matrix_t *)iMat->buffer, node->localDataPtr))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
		if (!status) {
			if (node->paramList[2]->u.mat.type != VX_TYPE_FLOAT32)
				return VX_ERROR_INVALID_TYPE;
			if (node->paramList[2]->u.mat.columns != 2 || node->paramList[2]->u.mat.rows != 3)
				return VX_ERROR_INVALID_DIMENSION;
			// output image dimensions have no constraints
			vx_meta_format meta;
			meta = &node->metaList[0];
			meta->data.u.img.width = node->paramList[0]->u.img.width;
			meta->data.u.img.height = node->paramList[0]->u.img.height;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		int alignedWidth = (node->paramList[0]->u.img.width + 15) & ~15;		// Next highest multiple of 16, so that the buffer is aligned for all three lines
		node->localDataSize = 2 * alignedWidth*sizeof(float);				// Three rows (+some extra) worth of scratch memory			
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		agoCodeGenOpenCL_BilinearSample(node->opencl_code);
		agoCodeGenOpenCL_BilinearSampleFXY(node->opencl_code);
		char textBuffer[4096];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s(U8x8 * r, uint x, uint y, __global uchar * p, uint stride, uint width, uint height, ago_affine_matrix_t matrix)\n"
			"{\n"
			"  U8x8 rv; float4 f;\n"
			"  float sx, sy;\n"
			"  float dx = (float)x, dy = (float)y;\n"
			"  sx = mad(dy, matrix.M[1][0], matrix.M[2][0]); sx = mad(dx, matrix.M[0][0], sx);\n"
			"  sy = mad(dy, matrix.M[1][1], matrix.M[2][1]); sy = mad(dx, matrix.M[0][1], sy);\n"
			"  f.s0 = BilinearSampleFXY(p, stride, sx, sy);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; f.s1 = BilinearSampleFXY(p, stride, sx, sy);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; f.s2 = BilinearSampleFXY(p, stride, sx, sy);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; f.s3 = BilinearSampleFXY(p, stride, sx, sy);\n"
			"  rv.s0 = amd_pack(f);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; f.s0 = BilinearSampleFXY(p, stride, sx, sy);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; f.s1 = BilinearSampleFXY(p, stride, sx, sy);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; f.s2 = BilinearSampleFXY(p, stride, sx, sy);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; f.s3 = BilinearSampleFXY(p, stride, sx, sy);\n"
			"  rv.s1 = amd_pack(f);\n"
			"  *r = rv;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
		node->opencl_type = NODE_OPENCL_TYPE_MEM2REG | NODE_OPENCL_TYPE_NEED_IMGSIZE;
		node->opencl_param_as_value_mask |= (1 << 2); // matrix parameter needs to be passed by value
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_WarpAffine_U8_U8_Bilinear_Constant(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iMat = node->paramList[2];
		if (HafCpu_WarpAffine_U8_U8_Bilinear_Constant(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
			iImg->u.img.width, iImg->u.img.height, iImg->buffer, iImg->u.img.stride_in_bytes, (ago_affine_matrix_t *)iMat->buffer, node->paramList[3]->u.scalar.u.u, node->localDataPtr))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
		if (!status) {
			if (node->paramList[2]->u.mat.type != VX_TYPE_FLOAT32)
				return VX_ERROR_INVALID_TYPE;
			if (node->paramList[2]->u.mat.columns != 2 || node->paramList[2]->u.mat.rows != 3)
				return VX_ERROR_INVALID_DIMENSION;
			if (node->paramList[3]->u.scalar.type != VX_TYPE_UINT8)
				return VX_ERROR_INVALID_FORMAT;
			// output image dimensions have no constraints
			vx_meta_format meta;
			meta = &node->metaList[0];
			meta->data.u.img.width = node->paramList[0]->u.img.width;
			meta->data.u.img.height = node->paramList[0]->u.img.height;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		int alignedWidth = (node->paramList[0]->u.img.width + 15) & ~15;		// Next highest multiple of 16, so that the buffer is aligned for all three lines
		node->localDataSize = 2 * alignedWidth*sizeof(float);				// Three rows (+some extra) worth of scratch memory			
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		agoCodeGenOpenCL_SampleWithConstBorder(node->opencl_code);
		agoCodeGenOpenCL_BilinearSample(node->opencl_code);
		agoCodeGenOpenCL_BilinearSampleWithConstBorder(node->opencl_code);
		agoCodeGenOpenCL_BilinearSampleFXYConstant(node->opencl_code);
		char textBuffer[4096];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s(U8x8 * r, uint x, uint y, __global uchar * p, uint stride, uint width, uint height, ago_affine_matrix_t matrix, uint borderValue)\n"
			"{\n"
			"  U8x8 rv; float4 f;\n"
			"  float sx, sy;\n"
			"  float dx = (float)x, dy = (float)y;\n"
			"  sx = mad(dy, matrix.M[1][0], matrix.M[2][0]); sx = mad(dx, matrix.M[0][0], sx);\n"
			"  sy = mad(dy, matrix.M[1][1], matrix.M[2][1]); sy = mad(dx, matrix.M[0][1], sy);\n"
			"  f.s0 = BilinearSampleFXYConstant(p, stride, width, height, sx, sy, borderValue);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; f.s1 = BilinearSampleFXYConstant(p, stride, width, height, sx, sy, borderValue);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; f.s2 = BilinearSampleFXYConstant(p, stride, width, height, sx, sy, borderValue);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; f.s3 = BilinearSampleFXYConstant(p, stride, width, height, sx, sy, borderValue);\n"
			"  rv.s0 = amd_pack(f);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; f.s0 = BilinearSampleFXYConstant(p, stride, width, height, sx, sy, borderValue);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; f.s1 = BilinearSampleFXYConstant(p, stride, width, height, sx, sy, borderValue);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; f.s2 = BilinearSampleFXYConstant(p, stride, width, height, sx, sy, borderValue);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; f.s3 = BilinearSampleFXYConstant(p, stride, width, height, sx, sy, borderValue);\n"
			"  rv.s1 = amd_pack(f);\n"
			"  *r = rv;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
		node->opencl_type = NODE_OPENCL_TYPE_MEM2REG | NODE_OPENCL_TYPE_NEED_IMGSIZE;
		node->opencl_param_as_value_mask |= (1 << 2); // matrix parameter needs to be passed by value
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_WarpPerspective_U8_U8_Nearest(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iMat = node->paramList[2];
		if (HafCpu_WarpPerspective_U8_U8_Nearest(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
			iImg->u.img.width, iImg->u.img.height, iImg->buffer, iImg->u.img.stride_in_bytes, (ago_perspective_matrix_t *)iMat->buffer, node->localDataPtr))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
		if (!status) {
			if (node->paramList[2]->u.mat.type != VX_TYPE_FLOAT32)
				return VX_ERROR_INVALID_TYPE;
			if (node->paramList[2]->u.mat.columns != 3 || node->paramList[2]->u.mat.rows != 3)
				return VX_ERROR_INVALID_DIMENSION;
			// output image dimensions have no constraints
			vx_meta_format meta;
			meta = &node->metaList[0];
			meta->data.u.img.width = node->paramList[0]->u.img.width;
			meta->data.u.img.height = node->paramList[0]->u.img.height;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		int alignedWidth = (node->paramList[0]->u.img.width + 15) & ~15;		// Next highest multiple of 16, so that the buffer is aligned for all three lines
		node->localDataSize = 3 * alignedWidth*sizeof(float);				// Three rows (+some extra) worth of scratch memory			
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		char textBuffer[4096];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s(U8x8 * r, uint x, uint y, __global uchar * p, uint stride, uint width, uint height, ago_perspective_matrix_t matrix)\n"
			"{\n"
			"  U8x8 rv;\n"
			"  float sx, sy, sz, isz;\n"
			"  float dx = (float)x, dy = (float)y;\n"
			"  sx = mad(dy, matrix.M[1][0], matrix.M[2][0]); sx = mad(dx, matrix.M[0][0], sx);\n"
			"  sy = mad(dy, matrix.M[1][1], matrix.M[2][1]); sy = mad(dx, matrix.M[0][1], sy);\n"
			"  sz = mad(dy, matrix.M[1][2], matrix.M[2][2]); sz = mad(dx, matrix.M[0][2], sz);\n"
			"  isz = 1.0f / sz; rv.s0 = p[mad24(stride, (uint)(sy*isz), (uint)(sx*isz))];\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; rv.s0 |= p[mad24(stride, (uint)(sy*isz), (uint)(sx*isz))] << 8;\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; rv.s0 |= p[mad24(stride, (uint)(sy*isz), (uint)(sx*isz))] << 16;\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; rv.s0 |= p[mad24(stride, (uint)(sy*isz), (uint)(sx*isz))] << 24;\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; rv.s1  = p[mad24(stride, (uint)(sy*isz), (uint)(sx*isz))];\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; rv.s1 |= p[mad24(stride, (uint)(sy*isz), (uint)(sx*isz))] << 8;\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; rv.s1 |= p[mad24(stride, (uint)(sy*isz), (uint)(sx*isz))] << 16;\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; rv.s1 |= p[mad24(stride, (uint)(sy*isz), (uint)(sx*isz))] << 24;\n"
			"  *r = rv;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
		node->opencl_type = NODE_OPENCL_TYPE_MEM2REG | NODE_OPENCL_TYPE_NEED_IMGSIZE;
		node->opencl_param_as_value_mask |= (1 << 2); // matrix parameter needs to be passed by value
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_WarpPerspective_U8_U8_Nearest_Constant(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iMat = node->paramList[2];
		if (HafCpu_WarpPerspective_U8_U8_Nearest_Constant(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
			iImg->u.img.width, iImg->u.img.height, iImg->buffer, iImg->u.img.stride_in_bytes, (ago_perspective_matrix_t *)iMat->buffer, node->paramList[3]->u.scalar.u.u, node->localDataPtr))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
		if (!status) {
			if (node->paramList[2]->u.mat.type != VX_TYPE_FLOAT32)
				return VX_ERROR_INVALID_TYPE;
			if (node->paramList[2]->u.mat.columns != 3 || node->paramList[2]->u.mat.rows != 3)
				return VX_ERROR_INVALID_DIMENSION;
			if (node->paramList[3]->u.scalar.type != VX_TYPE_UINT8)
				return VX_ERROR_INVALID_FORMAT;
			// output image dimensions have no constraints
			vx_meta_format meta;
			meta = &node->metaList[0];
			meta->data.u.img.width = node->paramList[0]->u.img.width;
			meta->data.u.img.height = node->paramList[0]->u.img.height;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		int alignedWidth = (node->paramList[0]->u.img.width + 15) & ~15;		// Next highest multiple of 16, so that the buffer is aligned for all three lines
		node->localDataSize = 3 * alignedWidth*sizeof(float);				// Three rows (+some extra) worth of scratch memory			
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		char textBuffer[4096];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s(U8x8 * r, uint x, uint y, __global uchar * p, uint stride, uint width, uint height, ago_perspective_matrix_t matrix, uint border)\n"
			"{\n"
			"  width -= 2; height -= 2;\n"
			"  U8x8 rv;\n"
			"  float sx, sy, sz, isz; uint mask, v;\n"
			"  float dx = (float)x, dy = (float)y;\n"
			"  sx = mad(dy, matrix.M[1][0], matrix.M[2][0]); sx = mad(dx, matrix.M[0][0], sx);\n"
			"  sy = mad(dy, matrix.M[1][1], matrix.M[2][1]); sy = mad(dx, matrix.M[0][1], sy);\n"
			"  sz = mad(dy, matrix.M[1][2], matrix.M[2][2]); sz = mad(dx, matrix.M[0][2], sz);\n"
			"  isz = 1.0f / sz; x = (uint)(int)(sx*isz); y = (uint)(int)(sy*isz);\n"
			"  mask = ((int)(x | (width - x) | y | (height - y))) >> 31; mask = ~mask;\n"
			"  x &= mask; y &= mask; v = p[mad24(stride, y, x)]; v = bitselect(border, v, mask); rv.s0 = v;\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; x = (uint)(int)(sx*isz); y = (uint)(int)(sy*isz);\n"
			"  mask = ((int)(x | (width - x) | y | (height - y))) >> 31; mask = ~mask;\n"
			"  x &= mask; y &= mask; v = p[mad24(stride, y, x)]; v = bitselect(border, v, mask); rv.s0 |= (v << 8);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; x = (uint)(int)(sx*isz); y = (uint)(int)(sy*isz);\n"
			"  mask = ((int)(x | (width - x) | y | (height - y))) >> 31; mask = ~mask;\n"
			"  x &= mask; y &= mask; v = p[mad24(stride, y, x)]; v = bitselect(border, v, mask); rv.s0 |= (v << 16);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; x = (uint)(int)(sx*isz); y = (uint)(int)(sy*isz);\n"
			"  mask = ((int)(x | (width - x) | y | (height - y))) >> 31; mask = ~mask;\n"
			"  x &= mask; y &= mask; v = p[mad24(stride, y, x)]; v = bitselect(border, v, mask); rv.s0 |= (v << 24);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; x = (uint)(int)(sx*isz); y = (uint)(int)(sy*isz);\n"
			"  mask = ((int)(x | (width - x) | y | (height - y))) >> 31; mask = ~mask;\n"
			"  x &= mask; y &= mask; v = p[mad24(stride, y, x)]; v = bitselect(border, v, mask); rv.s1 = v;\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; x = (uint)(int)(sx*isz); y = (uint)(int)(sy*isz);\n"
			"  mask = ((int)(x | (width - x) | y | (height - y))) >> 31; mask = ~mask;\n"
			"  x &= mask; y &= mask; v = p[mad24(stride, y, x)]; v = bitselect(border, v, mask); rv.s1 |= (v << 8);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; x = (uint)(int)(sx*isz); y = (uint)(int)(sy*isz);\n"
			"  mask = ((int)(x | (width - x) | y | (height - y))) >> 31; mask = ~mask;\n"
			"  x &= mask; y &= mask; v = p[mad24(stride, y, x)]; v = bitselect(border, v, mask); rv.s1 |= (v << 16);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; x = (uint)(int)(sx*isz); y = (uint)(int)(sy*isz);\n"
			"  mask = ((int)(x | (width - x) | y | (height - y))) >> 31; mask = ~mask;\n"
			"  x &= mask; y &= mask; v = p[mad24(stride, y, x)]; v = bitselect(border, v, mask); rv.s1 |= (v << 24);\n"
			"  *r = rv;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
		node->opencl_type = NODE_OPENCL_TYPE_MEM2REG | NODE_OPENCL_TYPE_NEED_IMGSIZE;
		node->opencl_param_as_value_mask |= (1 << 2); // matrix parameter needs to be passed by value
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_WarpPerspective_U8_U8_Bilinear(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iMat = node->paramList[2];
		if (HafCpu_WarpPerspective_U8_U8_Bilinear(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
			iImg->u.img.width, iImg->u.img.height, iImg->buffer, iImg->u.img.stride_in_bytes, (ago_perspective_matrix_t *)iMat->buffer, node->localDataPtr))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
		if (!status) {
			if (node->paramList[2]->u.mat.type != VX_TYPE_FLOAT32)
				return VX_ERROR_INVALID_TYPE;
			if (node->paramList[2]->u.mat.columns != 3 || node->paramList[2]->u.mat.rows != 3)
				return VX_ERROR_INVALID_DIMENSION;
			// output image dimensions have no constraints
			vx_meta_format meta;
			meta = &node->metaList[0];
			meta->data.u.img.width = node->paramList[0]->u.img.width;
			meta->data.u.img.height = node->paramList[0]->u.img.height;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		int alignedWidth = (node->paramList[0]->u.img.width + 15) & ~15;		// Next highest multiple of 16, so that the buffer is aligned for all three lines
		node->localDataSize = 3 * alignedWidth*sizeof(float);				// Three rows (+some extra) worth of scratch memory			
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		agoCodeGenOpenCL_BilinearSample(node->opencl_code);
		agoCodeGenOpenCL_BilinearSampleFXY(node->opencl_code);
		char textBuffer[4096];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s(U8x8 * r, uint x, uint y, __global uchar * p, uint stride, uint width, uint height, ago_perspective_matrix_t matrix)\n"
			"{\n"
			"  U8x8 rv; float4 f;\n"
			"  float sx, sy, sz, isz;\n"
			"  float dx = (float)x, dy = (float)y;\n"
			"  sx = mad(dy, matrix.M[1][0], matrix.M[2][0]); sx = mad(dx, matrix.M[0][0], sx);\n"
			"  sy = mad(dy, matrix.M[1][1], matrix.M[2][1]); sy = mad(dx, matrix.M[0][1], sy);\n"
			"  sz = mad(dy, matrix.M[1][2], matrix.M[2][2]); sz = mad(dx, matrix.M[0][2], sz);\n"
			"  isz = 1.0f / sz; f.s0 = BilinearSampleFXY(p, stride, sx*isz, sy*isz);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; f.s1 = BilinearSampleFXY(p, stride, sx*isz, sy*isz);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; f.s2 = BilinearSampleFXY(p, stride, sx*isz, sy*isz);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; f.s3 = BilinearSampleFXY(p, stride, sx*isz, sy*isz);\n"
			"  rv.s0 = amd_pack(f);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; f.s0 = BilinearSampleFXY(p, stride, sx*isz, sy*isz);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; f.s1 = BilinearSampleFXY(p, stride, sx*isz, sy*isz);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; f.s2 = BilinearSampleFXY(p, stride, sx*isz, sy*isz);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; f.s3 = BilinearSampleFXY(p, stride, sx*isz, sy*isz);\n"
			"  rv.s1 = amd_pack(f);\n"
			"  *r = rv;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
		node->opencl_type = NODE_OPENCL_TYPE_MEM2REG | NODE_OPENCL_TYPE_NEED_IMGSIZE;
		node->opencl_param_as_value_mask |= (1 << 2); // matrix parameter needs to be passed by value
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_WarpPerspective_U8_U8_Bilinear_Constant(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iMat = node->paramList[2];
		if (HafCpu_WarpPerspective_U8_U8_Bilinear_Constant(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
			iImg->u.img.width, iImg->u.img.height, iImg->buffer, iImg->u.img.stride_in_bytes, (ago_perspective_matrix_t *)iMat->buffer, node->paramList[3]->u.scalar.u.u, node->localDataPtr))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
		if (!status) {
			if (node->paramList[2]->u.mat.type != VX_TYPE_FLOAT32)
				return VX_ERROR_INVALID_TYPE;
			if (node->paramList[2]->u.mat.columns != 3 || node->paramList[2]->u.mat.rows != 3)
				return VX_ERROR_INVALID_DIMENSION;
			if (node->paramList[3]->u.scalar.type != VX_TYPE_UINT8)
				return VX_ERROR_INVALID_FORMAT;
			// output image dimensions have no constraints
			vx_meta_format meta;
			meta = &node->metaList[0];
			meta->data.u.img.width = node->paramList[0]->u.img.width;
			meta->data.u.img.height = node->paramList[0]->u.img.height;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		int alignedWidth = (node->paramList[0]->u.img.width + 15) & ~15;		// Next highest multiple of 16, so that the buffer is aligned for all three lines
		node->localDataSize = 3 * alignedWidth*sizeof(float);				// Three rows (+some extra) worth of scratch memory			
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		agoCodeGenOpenCL_SampleWithConstBorder(node->opencl_code);
		agoCodeGenOpenCL_BilinearSample(node->opencl_code);
		agoCodeGenOpenCL_BilinearSampleWithConstBorder(node->opencl_code);
		agoCodeGenOpenCL_BilinearSampleFXYConstant(node->opencl_code);
		char textBuffer[8192];
		sprintf(textBuffer, OPENCL_FORMAT(
			"void %s(U8x8 * r, uint x, uint y, __global uchar * p, uint stride, uint width, uint height, ago_perspective_matrix_t matrix, uint borderValue)\n"
			"{\n"
			"  U8x8 rv; float4 f;\n"
			"  float sx, sy, sz, isz;\n"
			"  float dx = (float)x, dy = (float)y;\n"
			"  sx = mad(dy, matrix.M[1][0], matrix.M[2][0]); sx = mad(dx, matrix.M[0][0], sx);\n"
			"  sy = mad(dy, matrix.M[1][1], matrix.M[2][1]); sy = mad(dx, matrix.M[0][1], sy);\n"
			"  sz = mad(dy, matrix.M[1][2], matrix.M[2][2]); sz = mad(dx, matrix.M[0][2], sz);\n"
			"  isz = 1.0f / sz; f.s0 = BilinearSampleFXYConstant(p, stride, width, height, sx*isz, sy*isz, borderValue);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; f.s1 = BilinearSampleFXYConstant(p, stride, width, height, sx*isz, sy*isz, borderValue);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; f.s2 = BilinearSampleFXYConstant(p, stride, width, height, sx*isz, sy*isz, borderValue);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; f.s3 = BilinearSampleFXYConstant(p, stride, width, height, sx*isz, sy*isz, borderValue);\n"
			"  rv.s0 = amd_pack(f);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; f.s0 = BilinearSampleFXYConstant(p, stride, width, height, sx*isz, sy*isz, borderValue);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; f.s1 = BilinearSampleFXYConstant(p, stride, width, height, sx*isz, sy*isz, borderValue);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; f.s2 = BilinearSampleFXYConstant(p, stride, width, height, sx*isz, sy*isz, borderValue);\n"
			"  sx += matrix.M[0][0]; sy += matrix.M[0][1]; sz += matrix.M[0][2]; isz = 1.0f / sz; f.s3 = BilinearSampleFXYConstant(p, stride, width, height, sx*isz, sy*isz, borderValue);\n"
			"  rv.s1 = amd_pack(f);\n"
			"  *r = rv;\n"
			"}\n"
			), node->opencl_name);
		node->opencl_code += textBuffer;
		node->opencl_type = NODE_OPENCL_TYPE_MEM2REG | NODE_OPENCL_TYPE_NEED_IMGSIZE;
		node->opencl_param_as_value_mask |= (1 << 2); // matrix parameter needs to be passed by value
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_ScaleImage_U8_U8_Nearest(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ScaleImage_U8_U8_Nearest(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
			iImg->u.img.width, iImg->u.img.height, iImg->buffer, iImg->u.img.stride_in_bytes, (AgoConfigScaleMatrix *)node->localDataPtr))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
		if (!status) {
			vx_meta_format meta;
			meta = &node->metaList[0];
			meta->data.u.img.width = node->paramList[0]->u.img.width;
			meta->data.u.img.height = node->paramList[0]->u.img.height;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		int alignedWidth = (oImg->u.img.width + 15) & ~15;
		int alignedHeight = (oImg->u.img.height + 15) & ~15;
		node->localDataSize = sizeof(AgoConfigScaleMatrix) + (alignedWidth * 2) + (alignedHeight * 2);
		node->localDataPtr = (vx_uint8 *)agoAllocMemory(node->localDataSize);
		if (!node->localDataPtr) return VX_ERROR_NO_MEMORY;
		// compute scale matrix from the input and output image sizes
		AgoConfigScaleMatrix * scalemat = (AgoConfigScaleMatrix *)node->localDataPtr;
		scalemat->xscale = (vx_float32)((vx_float64)iImg->u.img.width / (vx_float64)oImg->u.img.width);
		scalemat->yscale = (vx_float32)((vx_float64)iImg->u.img.height / (vx_float64)oImg->u.img.height);
		scalemat->xoffset = (vx_float32)((vx_float64)iImg->u.img.width / (vx_float64)oImg->u.img.width * 0.5);
		scalemat->yoffset = (vx_float32)((vx_float64)iImg->u.img.height / (vx_float64)oImg->u.img.height * 0.5);
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
		if (node->localDataPtr) {
			agoReleaseMemory(node->localDataPtr);
			node->localDataPtr = nullptr;
		}
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoConfigScaleMatrix scalemat; // compute scale matrix from the input and output image sizes
		scalemat.xscale = (vx_float32)((vx_float64)iImg->u.img.width / (vx_float64)oImg->u.img.width);
		scalemat.yscale = (vx_float32)((vx_float64)iImg->u.img.height / (vx_float64)oImg->u.img.height);
		scalemat.xoffset = (vx_float32)((vx_float64)iImg->u.img.width / (vx_float64)oImg->u.img.width * 0.5);
		scalemat.yoffset = (vx_float32)((vx_float64)iImg->u.img.height / (vx_float64)oImg->u.img.height * 0.5);
		char textBuffer[1024];
		sprintf(textBuffer,
			OPENCL_FORMAT(
			"void %s(U8x8 * r, uint x, uint y, __global uchar * p, uint stride)\n"
			"{\n"
			"  float4 scaleInfo = (float4)(%.12f,%.12f,%.12f,%.12f);\n"
			"  U8x8 rv;\n"
			"  p += stride*(uint)mad((float)y, scaleInfo.s1, scaleInfo.s3);\n"
			"  float fx = mad((float)x, scaleInfo.s0, scaleInfo.s2);\n"
			"  rv.s0  = p[(int)fx];\n"
			"  fx += scaleInfo.s0; rv.s0 |= p[(int)fx] << 8;\n"
			"  fx += scaleInfo.s0; rv.s0 |= p[(int)fx] << 16;\n"
			"  fx += scaleInfo.s0; rv.s0 |= p[(int)fx] << 24;\n"
			"  fx += scaleInfo.s0; rv.s1  = p[(int)fx];\n"
			"  fx += scaleInfo.s0; rv.s1 |= p[(int)fx] << 8;\n"
			"  fx += scaleInfo.s0; rv.s1 |= p[(int)fx] << 16;\n"
			"  fx += scaleInfo.s0; rv.s1 |= p[(int)fx] << 24;\n"
			"  *r = rv;\n"
			"}\n"
			), node->opencl_name, scalemat.xscale, scalemat.yscale, scalemat.xoffset, scalemat.yoffset);
		node->opencl_code += textBuffer;
		node->opencl_type = NODE_OPENCL_TYPE_MEM2REG;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * inp = node->paramList[0];
		AgoData * out = node->paramList[1];
		vx_float32 widthOut = (vx_float32)out->u.img.width;
		vx_float32 heightOut = (vx_float32)out->u.img.height;
		vx_float32 widthIn = (vx_float32)inp->u.img.width;
		vx_float32 heightIn = (vx_float32)inp->u.img.height;
		out->u.img.rect_valid.start_x = (vx_uint32)(((inp->u.img.rect_valid.start_x + 0.5f) * widthOut / widthIn) - 0.5f);
		out->u.img.rect_valid.start_y = (vx_uint32)(((inp->u.img.rect_valid.start_y + 0.5f) * heightOut / heightIn) - 0.5f);
		out->u.img.rect_valid.end_x = (vx_uint32)(((inp->u.img.rect_valid.end_x + 0.5f) * widthOut / widthIn) - 0.5f);
		out->u.img.rect_valid.end_y = (vx_uint32)(((inp->u.img.rect_valid.end_y + 0.5f) * heightOut / heightIn) - 0.5f);
	}
	return status;
}

int agoKernel_ScaleImage_U8_U8_Bilinear(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ScaleImage_U8_U8_Bilinear(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
			iImg->u.img.width, iImg->u.img.height, iImg->buffer, iImg->u.img.stride_in_bytes, (AgoConfigScaleMatrix *)node->localDataPtr))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
		if (!status) {
			vx_meta_format meta;
			meta = &node->metaList[0];
			meta->data.u.img.width = node->paramList[0]->u.img.width;
			meta->data.u.img.height = node->paramList[0]->u.img.height;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		int alignedWidth = (oImg->u.img.width + 15) & ~15;
		node->localDataSize = sizeof(AgoConfigScaleMatrix) + (alignedWidth * 6);
		node->localDataPtr = (vx_uint8 *)agoAllocMemory(node->localDataSize);
		if (!node->localDataPtr) return VX_ERROR_NO_MEMORY;
		// compute scale matrix from the input and output image sizes
		AgoConfigScaleMatrix * scalemat = (AgoConfigScaleMatrix *)node->localDataPtr;
		scalemat->xscale = (vx_float32)((vx_float64)iImg->u.img.width / (vx_float64)oImg->u.img.width);
		scalemat->yscale = (vx_float32)((vx_float64)iImg->u.img.height / (vx_float64)oImg->u.img.height);
		scalemat->xoffset = (vx_float32)((vx_float64)iImg->u.img.width / (vx_float64)oImg->u.img.width * 0.5 - 0.5);
		scalemat->yoffset = (vx_float32)((vx_float64)iImg->u.img.height / (vx_float64)oImg->u.img.height * 0.5 - 0.5);
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
		if (node->localDataPtr) {
			agoReleaseMemory(node->localDataPtr);
			node->localDataPtr = nullptr;
		}
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoConfigScaleMatrix scalemat; // compute scale matrix from the input and output image sizes
		scalemat.xscale = (vx_float32)((vx_float64)iImg->u.img.width / (vx_float64)oImg->u.img.width);
		scalemat.yscale = (vx_float32)((vx_float64)iImg->u.img.height / (vx_float64)oImg->u.img.height);
		scalemat.xoffset = (vx_float32)((vx_float64)iImg->u.img.width / (vx_float64)oImg->u.img.width * 0.5 - 0.5);
		scalemat.yoffset = (vx_float32)((vx_float64)iImg->u.img.height / (vx_float64)oImg->u.img.height * 0.5 - 0.5);
		agoCodeGenOpenCL_BilinearSample(node->opencl_code);
		agoCodeGenOpenCL_ScaleImage_U8_U8_Bilinear(node->opencl_code);
		char textBuffer[8192];
		sprintf(textBuffer,
			OPENCL_FORMAT(
			"void %s(U8x8 * r, uint x, uint y, __global uchar * p, uint stride)\n"
			"{\n"
			"  float4 scaleInfo = (float4)(%.12f,%.12f,%.12f,%.12f);\n"
			"  ScaleImage_U8_U8_Bilinear(r, x, y, p, stride, scaleInfo);"
			"}\n"
			), node->opencl_name, scalemat.xscale, scalemat.yscale, scalemat.xoffset, scalemat.yoffset);
		node->opencl_code += textBuffer;
		node->opencl_type = NODE_OPENCL_TYPE_MEM2REG;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * inp = node->paramList[0];
		AgoData * out = node->paramList[1];
		vx_float32 widthOut = (vx_float32)out->u.img.width;
		vx_float32 heightOut = (vx_float32)out->u.img.height;
		vx_float32 widthIn = (vx_float32)inp->u.img.width;
		vx_float32 heightIn = (vx_float32)inp->u.img.height;
		out->u.img.rect_valid.start_x = (vx_uint32)(((inp->u.img.rect_valid.start_x + 0.5f) * widthOut / widthIn) - 0.5f);
		out->u.img.rect_valid.start_y = (vx_uint32)(((inp->u.img.rect_valid.start_y + 0.5f) * heightOut / heightIn) - 0.5f);
		out->u.img.rect_valid.end_x = (vx_uint32)(((inp->u.img.rect_valid.end_x + 0.5f) * widthOut / widthIn) - 0.5f);
		out->u.img.rect_valid.end_y = (vx_uint32)(((inp->u.img.rect_valid.end_y + 0.5f) * heightOut / heightIn) - 0.5f);
	}
	return status;
}

int agoKernel_ScaleImage_U8_U8_Bilinear_Replicate(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ScaleImage_U8_U8_Bilinear_Replicate(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
			iImg->u.img.width, iImg->u.img.height, iImg->buffer, iImg->u.img.stride_in_bytes, (AgoConfigScaleMatrix *)node->localDataPtr))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
		if (!status) {
			vx_meta_format meta;
			meta = &node->metaList[0];
			meta->data.u.img.width = node->paramList[0]->u.img.width;
			meta->data.u.img.height = node->paramList[0]->u.img.height;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		int alignedWidth = (oImg->u.img.width + 15) & ~15;
		node->localDataSize = sizeof(AgoConfigScaleMatrix) + (alignedWidth * 6);
		node->localDataPtr = (vx_uint8 *)agoAllocMemory(node->localDataSize);
		if (!node->localDataPtr) return VX_ERROR_NO_MEMORY;
		// compute scale matrix from the input and output image sizes
		AgoConfigScaleMatrix * scalemat = (AgoConfigScaleMatrix *)node->localDataPtr;
		scalemat->xscale = (vx_float32)((vx_float64)iImg->u.img.width / (vx_float64)oImg->u.img.width);
		scalemat->yscale = (vx_float32)((vx_float64)iImg->u.img.height / (vx_float64)oImg->u.img.height);
		scalemat->xoffset = (vx_float32)((vx_float64)iImg->u.img.width / (vx_float64)oImg->u.img.width * 0.5 - 0.5);
		scalemat->yoffset = (vx_float32)((vx_float64)iImg->u.img.height / (vx_float64)oImg->u.img.height * 0.5 - 0.5);
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
		if (node->localDataPtr) {
			agoReleaseMemory(node->localDataPtr);
			node->localDataPtr = nullptr;
		}
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoConfigScaleMatrix scalemat; // compute scale matrix from the input and output image sizes
		scalemat.xscale = (vx_float32)((vx_float64)iImg->u.img.width / (vx_float64)oImg->u.img.width);
		scalemat.yscale = (vx_float32)((vx_float64)iImg->u.img.height / (vx_float64)oImg->u.img.height);
		scalemat.xoffset = (vx_float32)((vx_float64)iImg->u.img.width / (vx_float64)oImg->u.img.width * 0.5 - 0.5);
		scalemat.yoffset = (vx_float32)((vx_float64)iImg->u.img.height / (vx_float64)oImg->u.img.height * 0.5 - 0.5);
		agoCodeGenOpenCL_ClampPixelCoordinatesToBorder(node->opencl_code);
		agoCodeGenOpenCL_BilinearSample(node->opencl_code);
		agoCodeGenOpenCL_ScaleImage_U8_U8_Bilinear(node->opencl_code);
		char textBuffer[8192];
		sprintf(textBuffer,
			OPENCL_FORMAT(
			"void %s(U8x8 * r, uint x, uint y, __global uchar * p, uint stride, uint width, uint height)\n"
			"{\n"
			"  float4 scaleInfo = (float4)(%.12f,%.12f,%.12f,%.12f);\n"
			"  // compute source x, y coordinates\n"
			"  float fx = mad((float)x, scaleInfo.s0, scaleInfo.s2);\n"
			"  float fy = mad((float)y, scaleInfo.s1, scaleInfo.s3);\n"
			"  // check if all pixels stay within borders\n"
			"  if (fx >= 0.0f && fy >= 0.0f && mad(8.0f, scaleInfo.s0, fx) < (width - 1) && mad(1.0f, scaleInfo.s1, fy) < (height - 1)) {\n"
			"  	ScaleImage_U8_U8_Bilinear(r, x, y, p, stride, scaleInfo);\n"
			"  }\n"
			"  else {\n"
			"  	// compute x and y upper limits\n"
			"  	float fxlimit = (float)(width - 1), fylimit = (float)(height - 1);\n"
			"  	// compute y coordinate and y interpolation factors\n"
			"  	float fy0, fy1;\n"
			"  	fy0 = floor(fy); fy1 = fy - fy0; fy0 = 1.0f - fy1;\n"
			"  	// calculate sy and ystride\n"
			"  	uint2 ycoord = ClampPixelCoordinatesToBorder(fy, height - 1, stride);\n"
			"  	// process pixels\n"
			"  	p += mul24(ycoord.s0, stride);\n"
			"  	float frac;\n"
			"  	uint2 xcoord;\n"
			"  	uint xlimit = width - 1;\n"
			"  	U8x8 rv; float4 f;  xcoord = ClampPixelCoordinatesToBorder(fx, xlimit, 1); frac = fx - floor(fx); f.s0 = BilinearSample(p, ycoord.s1, xcoord.s1, fy0, fy1, xcoord.s0, 1.0f - frac, frac);\n"
			"  	fx += scaleInfo.s0; xcoord = ClampPixelCoordinatesToBorder(fx, xlimit, 1); frac = fx - floor(fx); f.s1 = BilinearSample(p, ycoord.s1, xcoord.s1, fy0, fy1, xcoord.s0, 1.0f - frac, frac);\n"
			"  	fx += scaleInfo.s0; xcoord = ClampPixelCoordinatesToBorder(fx, xlimit, 1); frac = fx - floor(fx); f.s2 = BilinearSample(p, ycoord.s1, xcoord.s1, fy0, fy1, xcoord.s0, 1.0f - frac, frac);\n"
			"  	fx += scaleInfo.s0; xcoord = ClampPixelCoordinatesToBorder(fx, xlimit, 1); frac = fx - floor(fx); f.s3 = BilinearSample(p, ycoord.s1, xcoord.s1, fy0, fy1, xcoord.s0, 1.0f - frac, frac);\n"
			"  	rv.s0 = amd_pack(f);\n"
			"  	fx += scaleInfo.s0; xcoord = ClampPixelCoordinatesToBorder(fx, xlimit, 1); frac = fx - floor(fx); f.s0 = BilinearSample(p, ycoord.s1, xcoord.s1, fy0, fy1, xcoord.s0, 1.0f - frac, frac);\n"
			"  	fx += scaleInfo.s0; xcoord = ClampPixelCoordinatesToBorder(fx, xlimit, 1); frac = fx - floor(fx); f.s1 = BilinearSample(p, ycoord.s1, xcoord.s1, fy0, fy1, xcoord.s0, 1.0f - frac, frac);\n"
			"  	fx += scaleInfo.s0; xcoord = ClampPixelCoordinatesToBorder(fx, xlimit, 1); frac = fx - floor(fx); f.s2 = BilinearSample(p, ycoord.s1, xcoord.s1, fy0, fy1, xcoord.s0, 1.0f - frac, frac);\n"
			"  	fx += scaleInfo.s0; xcoord = ClampPixelCoordinatesToBorder(fx, xlimit, 1); frac = fx - floor(fx); f.s3 = BilinearSample(p, ycoord.s1, xcoord.s1, fy0, fy1, xcoord.s0, 1.0f - frac, frac);\n"
			"  	rv.s1 = amd_pack(f);\n"
			"  	*r = rv;\n"
			"  }\n"
			"}\n"
			), node->opencl_name, scalemat.xscale, scalemat.yscale, scalemat.xoffset, scalemat.yoffset);
		node->opencl_code += textBuffer;
		node->opencl_type = NODE_OPENCL_TYPE_MEM2REG | NODE_OPENCL_TYPE_NEED_IMGSIZE;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_ScaleImage_U8_U8_Bilinear_Constant(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoData * iBorder = node->paramList[2];
		if (HafCpu_ScaleImage_U8_U8_Bilinear_Constant(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
			iImg->u.img.width, iImg->u.img.height, iImg->buffer, iImg->u.img.stride_in_bytes, (AgoConfigScaleMatrix *)node->localDataPtr, iBorder->u.scalar.u.u))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN_S(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8, VX_TYPE_UINT8);
		if (!status) {
			vx_meta_format meta;
			meta = &node->metaList[0];
			meta->data.u.img.width = node->paramList[0]->u.img.width;
			meta->data.u.img.height = node->paramList[0]->u.img.height;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		int alignedWidth = (oImg->u.img.width + 15) & ~15;
		node->localDataSize = sizeof(AgoConfigScaleMatrix) + (alignedWidth * 6) + (iImg->u.img.width+15)&~15;
		node->localDataPtr = (vx_uint8 *)agoAllocMemory(node->localDataSize);
		if (!node->localDataPtr) return VX_ERROR_NO_MEMORY;
		// compute scale matrix from the input and output image sizes
		AgoConfigScaleMatrix * scalemat = (AgoConfigScaleMatrix *)node->localDataPtr;
		scalemat->xscale = (vx_float32)((vx_float64)iImg->u.img.width / (vx_float64)oImg->u.img.width);
		scalemat->yscale = (vx_float32)((vx_float64)iImg->u.img.height / (vx_float64)oImg->u.img.height);
		scalemat->xoffset = (vx_float32)((vx_float64)iImg->u.img.width / (vx_float64)oImg->u.img.width * 0.5 - 0.5);
		scalemat->yoffset = (vx_float32)((vx_float64)iImg->u.img.height / (vx_float64)oImg->u.img.height * 0.5 - 0.5);
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
		if (node->localDataPtr) {
			agoReleaseMemory(node->localDataPtr);
			node->localDataPtr = nullptr;
		}
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		AgoConfigScaleMatrix scalemat; // compute scale matrix from the input and output image sizes
		scalemat.xscale = (vx_float32)((vx_float64)iImg->u.img.width / (vx_float64)oImg->u.img.width);
		scalemat.yscale = (vx_float32)((vx_float64)iImg->u.img.height / (vx_float64)oImg->u.img.height);
		scalemat.xoffset = (vx_float32)((vx_float64)iImg->u.img.width / (vx_float64)oImg->u.img.width * 0.5 - 0.5);
		scalemat.yoffset = (vx_float32)((vx_float64)iImg->u.img.height / (vx_float64)oImg->u.img.height * 0.5 - 0.5);
		agoCodeGenOpenCL_ClampPixelCoordinatesToBorder(node->opencl_code);
		agoCodeGenOpenCL_SampleWithConstBorder(node->opencl_code);
		agoCodeGenOpenCL_BilinearSampleWithConstBorder(node->opencl_code);
		agoCodeGenOpenCL_BilinearSample(node->opencl_code);
		agoCodeGenOpenCL_ScaleImage_U8_U8_Bilinear(node->opencl_code);
		char textBuffer[8192];
		sprintf(textBuffer,
			OPENCL_FORMAT(
			"void %s(U8x8 * r, uint x, uint y, __global uchar * p, uint stride, uint width, uint height, uint borderValue)\n"
			"{\n"
			"  float4 scaleInfo = (float4)(%.12f,%.12f,%.12f,%.12f);\n"
			"  // compute source x, y coordinates\n"
			"  float fx = mad((float)x, scaleInfo.s0, scaleInfo.s2);\n"
			"  float fy = mad((float)y, scaleInfo.s1, scaleInfo.s3);\n"
			"  // check if all pixels stay within borders\n"
			"  if (fx >= 0.0f && fy >= 0.0f && mad(8.0f, scaleInfo.s0, fx) < (width - 1) && mad(1.0f, scaleInfo.s1, fy) < (height - 1)) {\n"
			"  	ScaleImage_U8_U8_Bilinear(r, x, y, p, stride, scaleInfo);\n"
			"  }\n"
			"  else {\n"
			"  	// compute y coordinate interpolation factors\n"
			"  	float fy1 = fy - floor(fy);\n"
			"  	float fy0 = 1.0f - fy1;\n"
			"  	// compute pixel values\n"
			"  	int   sy = (int)floor(fy);\n"
			"  	float frac;\n"
			"  	U8x8 rv; float4 f;  frac = fx - floor(fx); f.s0 = BilinearSampleWithConstBorder(p, (int)floor(fx), sy, width, height, stride, 1.0f - frac, frac, fy0, fy1, borderValue);\n"
			"  	fx += scaleInfo.s0; frac = fx - floor(fx); f.s1 = BilinearSampleWithConstBorder(p, (int)floor(fx), sy, width, height, stride, 1.0f - frac, frac, fy0, fy1, borderValue);\n"
			"  	fx += scaleInfo.s0; frac = fx - floor(fx); f.s2 = BilinearSampleWithConstBorder(p, (int)floor(fx), sy, width, height, stride, 1.0f - frac, frac, fy0, fy1, borderValue);\n"
			"  	fx += scaleInfo.s0; frac = fx - floor(fx); f.s3 = BilinearSampleWithConstBorder(p, (int)floor(fx), sy, width, height, stride, 1.0f - frac, frac, fy0, fy1, borderValue);\n"
			"  	rv.s0 = amd_pack(f);\n"
			"  	fx += scaleInfo.s0; frac = fx - floor(fx); f.s0 = BilinearSampleWithConstBorder(p, (int)floor(fx), sy, width, height, stride, 1.0f - frac, frac, fy0, fy1, borderValue);\n"
			"  	fx += scaleInfo.s0; frac = fx - floor(fx); f.s1 = BilinearSampleWithConstBorder(p, (int)floor(fx), sy, width, height, stride, 1.0f - frac, frac, fy0, fy1, borderValue);\n"
			"  	fx += scaleInfo.s0; frac = fx - floor(fx); f.s2 = BilinearSampleWithConstBorder(p, (int)floor(fx), sy, width, height, stride, 1.0f - frac, frac, fy0, fy1, borderValue);\n"
			"  	fx += scaleInfo.s0; frac = fx - floor(fx); f.s3 = BilinearSampleWithConstBorder(p, (int)floor(fx), sy, width, height, stride, 1.0f - frac, frac, fy0, fy1, borderValue);\n"
			"  	rv.s1 = amd_pack(f);\n"
			"  	*r = rv;\n"
			"  }\n"
			"}\n"
			), node->opencl_name, scalemat.xscale, scalemat.yscale, scalemat.xoffset, scalemat.yoffset);
		node->opencl_code += textBuffer;
		node->opencl_type = NODE_OPENCL_TYPE_MEM2REG | NODE_OPENCL_TYPE_NEED_IMGSIZE;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_ScaleImage_U8_U8_Area(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_ScaleImage_U8_U8_Area(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
			iImg->u.img.width, iImg->u.img.height, iImg->buffer, iImg->u.img.stride_in_bytes, (AgoConfigScaleMatrix *)node->localDataPtr))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
		if (!status) {
			vx_meta_format meta;
			meta = &node->metaList[0];
			meta->data.u.img.width = node->paramList[0]->u.img.width;
			meta->data.u.img.height = node->paramList[0]->u.img.height;
		}
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		int alignedWidth = ((oImg->u.img.width + 15) & ~15) + ((iImg->u.img.width + 15) & ~15);
		node->localDataSize = sizeof(AgoConfigScaleMatrix) + alignedWidth * 2 + 16;
		node->localDataPtr = (vx_uint8 *)agoAllocMemory(node->localDataSize);
		if (!node->localDataPtr) return VX_ERROR_NO_MEMORY;
		// compute scale matrix from the input and output image sizes
		AgoConfigScaleMatrix * scalemat = (AgoConfigScaleMatrix *)node->localDataPtr;
		scalemat->xscale = (vx_float32)((vx_float64)iImg->u.img.width / (vx_float64)oImg->u.img.width);
		scalemat->yscale = (vx_float32)((vx_float64)iImg->u.img.height / (vx_float64)oImg->u.img.height);
		scalemat->xoffset = -0.5f;
		scalemat->yoffset = -0.5f;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
		if (node->localDataPtr) {
			agoReleaseMemory(node->localDataPtr);
			node->localDataPtr = nullptr;
		}
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		status = VX_SUCCESS;
		// compute configuration parameters
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		int dstWidth = oImg->u.img.width;
		int dstHeight = oImg->u.img.height;
		int srcWidth = iImg->u.img.width;
		int srcHeight = iImg->u.img.height;
		// generate code
		float Sx = (float)srcWidth / (float)dstWidth;
		float Sy = (float)srcHeight / (float)dstHeight;
		float fSx = Sx - floorf(Sx);
		float fSy = Sy - floorf(Sy);
		int Nx = (int)ceilf(Sx), Nxf = 0;
		int Ny = (int)ceilf(Sy), Nyf = 0;
		bool use_sad = (Nx % 4) ? false : true;
		if ((srcWidth % dstWidth) > 0) {
			use_sad = false;
			if ((dstWidth % (srcWidth % dstWidth)) > 0) Nxf++;
		}
		if ((srcHeight % dstHeight) > 0) {
			use_sad = false;
			if ((dstHeight % (srcHeight % dstHeight)) > 0) Nyf++;
		}
		bool need_align = ((Sx * 2.0f) != floorf(Sx * 2.0f)) ? true : false;
		std::string code;
		char item[1024];
		sprintf(item,
			OPENCL_FORMAT(
			"void %s(U8x8 * r, uint x, uint y, __global uchar * p, uint stride) // ScaleArea %gx%g using %dx%d window\n"
			"{\n"
			), node->opencl_name, Sx, Sy, Nx, Ny); code += item;
		if (fSx != 0.0f && fSy != 0.0f) {
			sprintf(item,
				OPENCL_FORMAT(
				"  float X = (float)x * %.12ff;\n"
				"  float Y = (float)y * %.12ff;\n"
				"  float fX = fract(X, &X);\n"
				"  float fY = fract(Y, &Y);\n"
				"  uint offset = stride * (int)Y + (int)X;\n"
				), Sx, Sy); code += item;
		}
		else if (fSx != 0.0f) {
			sprintf(item,
				OPENCL_FORMAT(
				"  float X = (float)x * %.12ff;\n"
				"  float fX = fract(X, &X);\n"
				"  uint offset = stride * (y * %d) + (int)X;\n"
				), Sx, Ny); code += item;
		}
		else if (fSy != 0.0f) {
			sprintf(item,
				OPENCL_FORMAT(
				"  float Y = (float)y * %.12ff;\n"
				"  float fY = fract(Y, &Y);\n"
				"  uint offset = stride * (int)Y + (x * %d);\n"
				), Sy, Nx); code += item;
		}
		else {
			sprintf(item,
				"  uint offset = stride * (y * %d) + (x * %d);\n"
				, Ny, Nx); code += item;
		}
		if (need_align) {
			code += "  uint align = offset & 3; offset -= align;\n";
		}
		code += "  p += offset;\n";
		if (fSy != 0.0f) {
			sprintf(item,
				OPENCL_FORMAT(
				"  F32x8 ftotal = (F32x8)0.0f;\n"
				"  float Sy = %.12ff, Syf = 1.0f - fY;\n"
				), Sy); code += item;
		}
		else if (use_sad) {
			code += "  U32x8 sum = (U32x8)0;\n";
		}
		else {
			code += "  F32x8 f = (F32x8)0.0f;\n";
		}
		sprintf(item,
			OPENCL_FORMAT(
			"  for (uint iy = 0; iy < %d; iy++) {\n"
			"    uint4 dw;\n"
			), Ny + Nyf); code += item;
		if (fSy != 0.0f) {
			code += "    F32x8 f = (F32x8)0.0f;\n";
		}
		if (fSx == 0.0f) {
			if (need_align) {
				for (int ix = 0, bpos = 0, lastdw = 0, nbytesprocessed = 0, jx = 0; ix < 8;) {
					int nbytes = 8 * Nx - nbytesprocessed;
					if (nbytes > 16) nbytes = 16;
					if (bpos > 0 && nbytes > 12) nbytes = 12;
					int ndw = (nbytes + 3) >> 2;
					char slist[] = "0123"; slist[ndw] = '\0';
					char vload[] = "vloadn"; vload[5] = ndw > 1 ? ('0' + ndw) : 0;
					sprintf(item, "    dw.s%s = %s(0, (__global uint *)&p[%d]);\n", slist, vload, bpos); code += item;
					for (int idw = 0, ldw = lastdw, jdw = lastdw ? 0 : 1; idw < ndw - (lastdw ? 0 : 1); idw++, jdw++) {
						sprintf(item, "    dw.s%d = amd_bytealign(dw.s%d, dw.s%d, align);\n", ldw, jdw, ldw); code += item;
						for (int jj = 0; jj < 4 && (nbytesprocessed + jj < 8 * Nx); jj++) {
							sprintf(item, "    f.s%d += amd_unpack%d(dw.s%d);\n", ix, jj, ldw); code += item;
							if (((nbytesprocessed + jj) % Nx) == (Nx - 1)) ix++;
						}
						// update for next iteration
						ldw = lastdw ? idw : idw + 1;
						nbytesprocessed += 4;
					}
					// update for next iteration
					bpos += nbytes;
					lastdw = ndw - 1;
				}
			}
			else {
				for (int ix = 0, bpos = 0; ix < 8;) {
					sprintf(item, "    dw = *((__global uint4 *)&p[%d]);\n", bpos); code += item;
					for (int jj = 0; jj < 4; jj++) {
						if (use_sad) {
							sprintf(item, "    sum.s%d = amd_sad(dw.s%d, 0u, sum.s%d);\n", ix, jj, ix); code += item;
							bpos += 4;
							if ((bpos % Nx) == 0) ix++;
						}
						else {
							for (int k = 0; k < 4 && ix < 8; k++) {
								sprintf(item, "    f.s%d += amd_unpack%d(dw.s%d);\n", ix, k, jj); code += item;
								bpos += 1;
								if ((bpos % Nx) == 0) ix++;
							}
						}
					}
				}
			}
		}
		else if ((Sx * 8.0f) == floorf(Sx * 8.0f)) {
			int nbytes = (int)(Sx * 8.0f);
			float factorOffset = 0.0f, factorRemaining = Sx;
			int xpos = 0;
			for (int offset = 0, bpos = 0, ix = 0, lastdw = 0; offset < nbytes;) {
				int N = nbytes - offset + (need_align && !lastdw ? 4 : 0);
				if (N > 16) N = 16;
				if (need_align && offset > 0 && N > 12) N = 12;
				int ndw = (N + 3) >> 2;
				char slist[] = "0123"; slist[ndw] = '\0';
				char vload[] = "vloadn"; vload[5] = ndw > 1 ? ('0' + ndw) : 0;
				sprintf(item, "    dw.s%s = %s(0, (__global uint *)&p[%d]);\n", slist, vload, bpos); code += item;
				if (need_align) {
					if (bpos == 0) bpos += 4;
					ndw -= (lastdw ? 0 : 1);
					for (int idw = 0, ldw = lastdw, jdw = lastdw ? 0 : 1; idw < ndw; idw++, jdw++) {
						sprintf(item, "    dw.s%d = amd_bytealign(dw.s%d, dw.s%d, align);\n", ldw, jdw, ldw); code += item;
						slist[idw] = '0' + ldw;
						// update for next iteration
						ldw = lastdw ? idw : idw + 1;
					}
					lastdw = ndw - (lastdw ? 1 : 0);
				}
				for (int jj = 0; jj < ndw; jj++, offset += 4, bpos += 4) {
					int jjdw = slist[jj] - '0';
					for (int k = 0; k < 4 && ix < 8;) {
						if (factorOffset == floorf(factorOffset)) {
							if (factorRemaining >= 1.0f) {
								sprintf(item, "    f.s%d += amd_unpack%d(dw.s%d);\n", ix, k, jjdw);
								factorOffset += 1.0f;
								factorRemaining -= 1.0f;
								k++;
							}
							else {
								sprintf(item, "    f.s%d += amd_unpack%d(dw.s%d) * %.12ff;\n", ix, k, jjdw, factorRemaining);
								factorOffset += factorRemaining;
								factorRemaining = 0.0f;
							}
						}
						else {
							float factorOffsetRemain = factorOffset - floorf(factorOffset);
							if ((factorOffsetRemain + factorRemaining) >= 1.0f) {
								float factor = 1.0f - factorOffsetRemain;
								sprintf(item, "    f.s%d += amd_unpack%d(dw.s%d) * %.12ff;\n", ix, k, jjdw, factor);
								factorOffset += factor;
								factorRemaining -= factor;
								k++;
							}
							else {
								sprintf(item, "    f.s%d += amd_unpack%d(dw.s%d) * %.12ff;\n", ix, k, jjdw, factorRemaining);
								factorOffset += factorRemaining;
								factorRemaining = 0.0f;
							}
						}
						code += item;
						if (factorRemaining <= 0.0f) {
							factorRemaining = Sx;
							ix++;
						}
					}
				}
			}
		}
		else {
			code += "    float Xs = fX, factor, Xi, Xf;\n";
			code += "    uint offset, align;\n";
			for (int ix = 0; ix < 8; ix++) {
				code += "    Xf = fract(Xs, &Xi); offset = (uint)Xi; align = offset & 3; offset -= align;";
				if (ix < 7) {
					sprintf(item, " Xs += %.12ff;", Sx); code += item;
				}
				code += "\n";
				int N = Nx + Nxf;
				if (N > 12) {
					status = VX_ERROR_NOT_SUPPORTED;
					agoAddLogEntry((vx_reference)node, status, "ERROR: ScalarArea OCL Nx+Nxf=%d not supported yet\n", N);
					return status;
				}
				int ndw = (N + 4 + 3) >> 2;
				char slist[] = "0123"; slist[ndw] = '\0';
				char vload[] = "vloadn"; vload[5] = ndw > 1 ? ('0' + ndw) : 0;
				sprintf(item, "    dw.s%s = %s(0, (__global uint *)&p[offset]);\n", slist, vload); code += item;
				for (int idw = 0; idw < ndw - 1; idw++) {
					sprintf(item, "    dw.s%d = amd_bytealign(dw.s%d, dw.s%d, align);\n", idw, idw + 1, idw); code += item;
				}
				int i = 0;
				sprintf(item, "    f.s%d += amd_unpack%d(dw.s%d) * (1.0f - Xf);\n", ix, i & 3, i >> 2); code += item;
				for (i = 1; i < Nx - 1; i++) {
					sprintf(item, "    f.s%d += amd_unpack%d(dw.s%d);\n", ix, i & 3, i >> 2); code += item;
				}
				sprintf(item, "    factor = %.12ff + Xf;", Sx - (Nx - 1)); code += item;
				sprintf(item, " f.s%d += amd_unpack%d(dw.s%d) * clamp(factor, 0.0f, 1.0f) +", ix, i & 3, i >> 2); i++; code += item;
				sprintf(item, " amd_unpack%d(dw.s%d) * clamp(factor-1.0f, 0.0f, 1.0f);\n", i & 3, i >> 2); code += item;
			}
		}
		if (fSy != 0.0f) {
			code += 
				OPENCL_FORMAT(
				"    f *= Syf;\n"
				"    ftotal += f;\n"
				"    Sy -= Syf;\n"
				"    Syf = clamp(Sy, 0.0f, 1.0f);\n"
				);
		}
		code +=
			"    p += stride;\n"
			"  }\n";
		if (use_sad) {
			code +=
				"  F32x8 f = convert_float8(sum);\n";
		}
		const char * fvar = (fSy != 0.0f) ? "ftotal" : "f";
		sprintf(item,
			OPENCL_FORMAT(
			"  %s *= %.12lff;\n"
			"  U8x8 rv;\n"
			"  rv.s0 = amd_pack(%s.s0123);\n"
			"  rv.s1 = amd_pack(%s.s4567);\n"
			"  *r = rv;\n"
			"}\n"
			), fvar, 1.0 / (double)(Sx*Sy), fvar, fvar); code += item;
		// save the OpenCL program code
		node->opencl_code += code;
		node->opencl_type = NODE_OPENCL_TYPE_MEM2REG;
	}
#endif
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
#if ENABLE_OPENCL                    
                    | AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_M2R
#endif                 
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * inp = node->paramList[0];
		AgoData * out = node->paramList[1];
		vx_float32 widthOut = (vx_float32)out->u.img.width;
		vx_float32 heightOut = (vx_float32)out->u.img.height;
		vx_float32 widthIn = (vx_float32)inp->u.img.width;
		vx_float32 heightIn = (vx_float32)inp->u.img.height;
		out->u.img.rect_valid.start_x = (vx_uint32)(((inp->u.img.rect_valid.start_x + 0.5f) * widthOut / widthIn) - 0.5f);
		out->u.img.rect_valid.start_y = (vx_uint32)(((inp->u.img.rect_valid.start_y + 0.5f) * heightOut / heightIn) - 0.5f);
		out->u.img.rect_valid.end_x = (vx_uint32)(((inp->u.img.rect_valid.end_x + 0.5f) * widthOut / widthIn) - 0.5f);
		out->u.img.rect_valid.end_y = (vx_uint32)(((inp->u.img.rect_valid.end_y + 0.5f) * heightOut / heightIn) - 0.5f);
	}
	return status;
}

int agoKernel_OpticalFlowPyrLK_XY_XY(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * newXY = node->paramList[0];
		AgoData * oldPyr = node->paramList[1];
		AgoData * newPyr = node->paramList[2];
		AgoData * oldXY = node->paramList[3];
		AgoData * newXYest = node->paramList[4];
		vx_enum    termination = node->paramList[5]->u.scalar.u.e;
		vx_float32 epsilon = node->paramList[6]->u.scalar.u.f;
		vx_uint32  num_iterations = node->paramList[7]->u.scalar.u.u;
		vx_bool    use_initial_estimate = node->paramList[8]->u.scalar.u.i ? vx_true_e : vx_false_e;
		vx_int32   window_dimension = (vx_int32)node->paramList[9]->u.scalar.u.s;
		ago_pyramid_u8_t *pPyrBuff = (ago_pyramid_u8_t *)oldPyr->buffer;
		if (oldXY->u.arr.numitems != newXYest->u.arr.numitems || oldXY->u.arr.numitems > newXY->u.arr.capacity) {
			status = VX_ERROR_INVALID_DIMENSION;
		}
		else if (HafCpu_OpticalFlowPyrLK_XY_XY_Generic((vx_keypoint_t *)newXY->buffer, oldPyr->u.pyr.scale, (vx_uint32)oldPyr->u.pyr.levels, (ago_pyramid_u8_t *)oldPyr->buffer,
			(ago_pyramid_u8_t *)newPyr->buffer, (vx_uint32)newXYest->u.arr.numitems, (vx_keypoint_t *)oldXY->buffer, (vx_keypoint_t *)newXYest->buffer,
			termination, epsilon, num_iterations, use_initial_estimate, pPyrBuff->width * 4, node->localDataPtr, window_dimension))
		{
			status = VX_FAILURE;
		}
		else {
			newXY->u.arr.numitems = oldXY->u.arr.numitems;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_OpticalFlowPyrLK_XY_XY(node);
		if (!status) {
			if (node->paramList[9]->u.scalar.type != VX_TYPE_SIZE) {
				status = VX_ERROR_INVALID_TYPE;
			}
			else {
				vx_size window_dimension = node->paramList[9]->u.scalar.u.s;
				if (window_dimension < 3 || window_dimension > AGO_OPTICALFLOWPYRLK_MAX_DIM) {
					status = VX_ERROR_INVALID_VALUE;
				}
			}
		}
	}
	else if (cmd == ago_kernel_cmd_initialize){
		// allocate pyramid images for storing scharr output
		AgoData * oldPyr = node->paramList[1];
		ago_pyramid_u8_t *pPyrBuff = (ago_pyramid_u8_t *)oldPyr->buffer;
		AgoData * newXYest = node->paramList[3];
		int pyrWidth = pPyrBuff[0].width;
		node->localDataSize = ((pPyrBuff->height*pPyrBuff->width * 4) + newXYest->u.arr.capacity*sizeof(ago_keypoint_t) + 256) + ((pyrWidth + 2) * 4 + 64);		// same as level 0 buffer; will be reused for lower levels. The second term, temp buffer for scharr
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
			| AGO_KERNEL_FLAG_DEVICE_CPU
			;
		status = VX_SUCCESS;
	}
	return status;
}

int agoKernel_OpticalFlowPrepareLK_XY_XY(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		// TBD
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// TBD
	}
	else if (cmd == ago_kernel_cmd_initialize){
		// TBD
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		// TBD
	}
	else if (cmd == ago_kernel_cmd_query_target_support) {
		// TBD
	}
	return status;
}

int agoKernel_OpticalFlowImageLK_XY_XY(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		// TBD
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// TBD
	}
	else if (cmd == ago_kernel_cmd_initialize){
		// TBD
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		// TBD
	}
	else if (cmd == ago_kernel_cmd_query_target_support) {
		// TBD
	}
	return status;
}

int agoKernel_OpticalFlowFinalLK_XY_XY(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		// TBD
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// TBD
	}
	else if (cmd == ago_kernel_cmd_initialize){
		// TBD
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		// TBD
	}
	else if (cmd == ago_kernel_cmd_query_target_support) {
		// TBD
	}
	return status;
}

int agoKernel_HarrisMergeSortAndPick_XY_HVC(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oXY = node->paramList[0];
		AgoData * oNum = node->paramList[1];
		AgoData * iImg = node->paramList[2];
		vx_float32 min_distance = node->paramList[3]->u.scalar.u.f;
		vx_uint32 cornerCount = 0;
		if (HafCpu_HarrisMergeSortAndPick_XY_HVC((vx_uint32)oXY->u.arr.capacity, (vx_keypoint_t *)oXY->buffer, &cornerCount, 
			iImg->u.img.width, iImg->u.img.height, (vx_float32 *)iImg->buffer, iImg->u.img.stride_in_bytes, min_distance)) {
			status = VX_FAILURE;
		}
		else {
			oXY->u.arr.numitems = min(cornerCount, (vx_uint32)oXY->u.arr.capacity);
			if (oNum) {
				oNum->u.scalar.u.s = cornerCount;
			}
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		if (node->paramList[2]->u.img.format != VX_DF_IMAGE_F32_AMD)
			return VX_ERROR_INVALID_FORMAT;
		else if (!node->paramList[2]->u.img.width || !node->paramList[2]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		else if (node->paramList[3]->u.scalar.type != VX_TYPE_FLOAT32)
			return VX_ERROR_INVALID_TYPE;
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.arr.itemtype = VX_TYPE_KEYPOINT;
		meta = &node->metaList[1];
		meta->data.u.scalar.type = VX_TYPE_SIZE;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_HarrisMergeSortAndPick_XY_XYS(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oXY = node->paramList[0];
		AgoData * oNum = node->paramList[1];
		AgoData * iXYS = node->paramList[2];
		vx_float32 min_distance = node->paramList[3]->u.scalar.u.f;
		ago_harris_grid_header_t * gridInfo = (ago_harris_grid_header_t *)node->localDataPtr;
		ago_coord2d_short_t * gridBuf = (ago_coord2d_short_t *)(node->localDataPtr ? &node->localDataPtr[sizeof(ago_harris_grid_header_t)] : nullptr);
		vx_uint32 cornerCount = 0;
		if (HafCpu_HarrisMergeSortAndPick_XY_XYS((vx_uint32)oXY->u.arr.capacity, (vx_keypoint_t *)oXY->buffer, &cornerCount,
			(ago_keypoint_xys_t *)iXYS->buffer, (vx_uint32)iXYS->u.arr.numitems, min_distance, gridInfo, gridBuf)) {
			status = VX_FAILURE;
		}
		else {
			oXY->u.arr.numitems = min(cornerCount, (vx_uint32)oXY->u.arr.capacity);
			if (oNum) {
				oNum->u.scalar.u.s = cornerCount;
			}
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		if (node->paramList[2]->u.arr.itemtype != AGO_TYPE_KEYPOINT_XYS)
			return VX_ERROR_INVALID_FORMAT;
		else if (node->paramList[3]->u.scalar.type != VX_TYPE_FLOAT32)
			return VX_ERROR_INVALID_TYPE;
		else if (node->paramList[3]->u.scalar.u.f <= 0.0f)
			return VX_ERROR_INVALID_VALUE;
		else if (node->paramList[4] && node->paramList[4]->u.scalar.type != VX_TYPE_UINT32)
			return VX_ERROR_INVALID_TYPE;
		else if (node->paramList[5] && node->paramList[5]->u.scalar.type != VX_TYPE_UINT32)
			return VX_ERROR_INVALID_TYPE;
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.arr.itemtype = VX_TYPE_KEYPOINT;
		meta = &node->metaList[1];
		meta->data.u.scalar.type = VX_TYPE_SIZE;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize) {
		vx_float32 min_distance = node->paramList[3]->u.scalar.u.f;
		if (min_distance > 2.0f) { // no need to check neighorhood when min_distance <= 2.0f
			// allocate a local buffer for a grid buffer with grid meta data
			vx_uint32 width = node->paramList[4]->u.scalar.u.u;
			vx_uint32 height = node->paramList[5]->u.scalar.u.u;
			vx_uint32 cellSize = (vx_uint32)floor(min_distance / M_SQRT2);
			vx_uint32 gridWidth = (width + cellSize - 1) / cellSize;
			vx_uint32 gridHeight = (height + cellSize - 1) / cellSize;
			vx_uint32 gridBufSize = (vx_uint32)(gridWidth * gridHeight * sizeof(ago_coord2d_short_t));
			node->localDataSize = sizeof(ago_harris_grid_header_t) + gridBufSize;
			node->localDataPtr = (vx_uint8 *)agoAllocMemory(node->localDataSize); if (!node->localDataPtr) return VX_ERROR_NO_MEMORY;
			ago_harris_grid_header_t * gridInfo = (ago_harris_grid_header_t *)node->localDataPtr;
			gridInfo->width = gridWidth;
			gridInfo->height = gridHeight;
			gridInfo->cellSize = cellSize;
			gridInfo->gridBufSize = gridBufSize;
		}
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_shutdown) {
		if (node->localDataPtr) {
			agoReleaseMemory(node->localDataPtr);
			node->localDataPtr = nullptr;
		}
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
			| AGO_KERNEL_FLAG_DEVICE_CPU
			;
		status = VX_SUCCESS;
	}
	return status;
}

int agoKernel_FastCornerMerge_XY_XY(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oXY = node->paramList[0];
		vx_keypoint_t * srcCorners[AGO_MAX_PARAMS] = { 0 };
		vx_uint32 srcCornerCount[AGO_MAX_PARAMS] = { 0 };
		vx_uint32 numSrcCornerBuffers = 0;
		for (vx_uint32 i = 1, j = 0; i < node->paramCount; i++) {
			if (node->paramList[i] && node->paramList[i]->u.arr.numitems) {
				srcCorners[numSrcCornerBuffers] = (vx_keypoint_t *)node->paramList[i]->buffer;
				srcCornerCount[numSrcCornerBuffers] = (vx_uint32)node->paramList[i]->u.arr.numitems;
				numSrcCornerBuffers++;
			}
		}
		vx_uint32 cornerCount = 0;
		if (HafCpu_FastCornerMerge_XY_XY((vx_uint32)oXY->u.arr.capacity, (vx_keypoint_t *)oXY->buffer, &cornerCount, numSrcCornerBuffers, srcCorners, srcCornerCount)) {
			status = VX_FAILURE;
		}
		else {
			oXY->u.arr.numitems = min(cornerCount, (vx_uint32)oXY->u.arr.capacity);
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		for (vx_uint32 i = 1; i < node->paramCount; i++) {
			if (node->paramList[i] && node->paramList[i]->u.arr.itemtype != VX_TYPE_KEYPOINT)
				return VX_ERROR_INVALID_TYPE;
		}
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.arr.itemtype = VX_TYPE_KEYPOINT;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_CannyEdgeTrace_U8_U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iStack = node->paramList[1];
		if (HafCpu_CannyEdgeTrace_U8_U8(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
										iStack->u.cannystack.count, (ago_coord2d_ushort_t *)iStack->buffer))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!node->paramList[0]->u.img.width || !node->paramList[0]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
        status = VX_SUCCESS;
	}
	return status;
}

int agoKernel_CannyEdgeTrace_U8_U8XY(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iStack = node->paramList[1];
		if (HafCpu_CannyEdgeTrace_U8_U8XY(oImg->u.img.width, oImg->u.img.height, oImg->buffer, oImg->u.img.stride_in_bytes,
										  iStack->u.cannystack.count, (ago_coord2d_ushort_t *)iStack->buffer, iStack->u.cannystack.stackTop))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		if (node->paramList[0]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!node->paramList[0]->u.img.width || !node->paramList[0]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
        status = VX_SUCCESS;
	}
	return status;
}

int agoKernel_IntegralImage_U32_U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oImg = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_IntegralImage_U32_U8(oImg->u.img.width, oImg->u.img.height, (vx_uint32 *)oImg->buffer, oImg->u.img.stride_in_bytes,
										iImg->buffer, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1OUT_1IN(node, VX_DF_IMAGE_U32, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		AgoData * out = node->paramList[0];
		AgoData * inp = node->paramList[1];
		out->u.img.rect_valid.start_x = inp->u.img.rect_valid.start_x;
		out->u.img.rect_valid.start_y = inp->u.img.rect_valid.start_y;
		out->u.img.rect_valid.end_x = inp->u.img.rect_valid.end_x;
		out->u.img.rect_valid.end_y = inp->u.img.rect_valid.end_y;
	}
	return status;
}

int agoKernel_Histogram_DATA_U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oDist = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		vx_uint32 numbins = (vx_uint32) oDist->u.dist.numbins;
		vx_uint32 offset = (vx_uint32)oDist->u.dist.offset;
		vx_uint32 range = (vx_uint32)oDist->u.dist.range;
		vx_uint32 window = oDist->u.dist.window;
		vx_uint32 * histOut = (vx_uint32 *)oDist->buffer;
		if (HafCpu_HistogramFixedBins_DATA_U8(histOut, numbins, offset, range, window, iImg->u.img.width, iImg->u.img.height, iImg->buffer, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1IN(node, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_MeanStdDev_DATA_U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oData = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_MeanStdDev_DATA_U8(&((ago_meanstddev_data_t *)oData->buffer)->sum, &((ago_meanstddev_data_t *)oData->buffer)->sumSquared, 
			iImg->u.img.rect_valid.end_x - iImg->u.img.rect_valid.start_x, iImg->u.img.rect_valid.end_y - iImg->u.img.rect_valid.start_y,
			iImg->buffer + (iImg->u.img.rect_valid.start_y*iImg->u.img.stride_in_bytes) + iImg->u.img.rect_valid.start_x, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
		else {
			((ago_meanstddev_data_t *)oData->buffer)->sampleCount = (iImg->u.img.rect_valid.end_x - iImg->u.img.rect_valid.start_x) * (iImg->u.img.rect_valid.end_y - iImg->u.img.rect_valid.start_y);
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1IN(node, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_MinMax_DATA_U8(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oData = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_MinMax_DATA_U8(&((ago_minmaxloc_data_t *)oData->buffer)->min, &((ago_minmaxloc_data_t *)oData->buffer)->max,
			iImg->u.img.rect_valid.end_x - iImg->u.img.rect_valid.start_x, iImg->u.img.rect_valid.end_y - iImg->u.img.rect_valid.start_y, 
			iImg->buffer + (iImg->u.img.rect_valid.start_y*iImg->u.img.stride_in_bytes) + iImg->u.img.rect_valid.start_x, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1IN(node, VX_DF_IMAGE_U8);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_MinMax_DATA_S16(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oData = node->paramList[0];
		AgoData * iImg = node->paramList[1];
		if (HafCpu_MinMax_DATA_S16(&((ago_minmaxloc_data_t *)oData->buffer)->min, &((ago_minmaxloc_data_t *)oData->buffer)->max,
			iImg->u.img.rect_valid.end_x - iImg->u.img.rect_valid.start_x, iImg->u.img.rect_valid.end_y - iImg->u.img.rect_valid.start_y, 
			(vx_int16 *)(iImg->buffer + (iImg->u.img.rect_valid.start_y*iImg->u.img.stride_in_bytes)) + iImg->u.img.rect_valid.start_x, iImg->u.img.stride_in_bytes)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = ValidateArguments_Img_1IN(node, VX_DF_IMAGE_S16);
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_Equalize_DATA_DATA(AgoNode * node, AgoKernelCommand cmd)
{
	// INFO: use VX_KERNEL_AMD_LUT_U8_U8 kernel
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oLut = node->paramList[0];
		AgoData * iDist = node->paramList[1];
		if (HafCpu_Equalize_DATA_DATA(oLut->buffer, 1, (vx_uint32 **)&iDist->buffer)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		if (node->paramList[0]->u.lut.type != VX_TYPE_UINT8)
			return VX_ERROR_INVALID_FORMAT;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_HistogramMerge_DATA_DATA(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		AgoData * oDist = node->paramList[0];
		vx_uint32 * srcDist[AGO_MAX_PARAMS];
		vx_uint32 numSrcDist = 0;
		for (vx_uint32 i = 1; i < node->paramCount; i++) {
			if (node->paramList[i]) {
				srcDist[numSrcDist++] = (vx_uint32 *)node->paramList[i]->buffer;
			}
		}
		if (HafCpu_HistogramMerge_DATA_DATA((vx_uint32 *)oDist->buffer, numSrcDist, srcDist)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		for (vx_uint32 i = 1; i < node->paramCount; i++) {
			if (node->paramList[i] && node->paramList[i]->u.arr.itemtype != VX_TYPE_KEYPOINT)
				return VX_ERROR_INVALID_TYPE;
		}
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.arr.itemtype = VX_TYPE_KEYPOINT;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_MeanStdDevMerge_DATA_DATA(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		vx_uint32	  totalSampleCount = 0;
		vx_uint32     numPartitions = 0;
		vx_float32    partSum[AGO_MAX_PARAMS];
		vx_float32    partSumOfSquared[AGO_MAX_PARAMS];
		for (vx_uint32 i = 2; i < node->paramCount; i++) {
			if (node->paramList[i]) {
				totalSampleCount += ((ago_meanstddev_data_t *)node->paramList[i]->buffer)->sampleCount;
				partSum[numPartitions] = ((ago_meanstddev_data_t *)node->paramList[i]->buffer)->sum;
				partSumOfSquared[numPartitions] = ((ago_meanstddev_data_t *)node->paramList[i]->buffer)->sumSquared;
				numPartitions++;
			}
		}
		if (HafCpu_MeanStdDevMerge_DATA_DATA(&node->paramList[0]->u.scalar.u.f, &node->paramList[1]->u.scalar.u.f, totalSampleCount, numPartitions, partSum, partSumOfSquared)) {
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.scalar.type = VX_TYPE_FLOAT32;
		meta = &node->metaList[1];
		meta->data.u.scalar.type = VX_TYPE_FLOAT32;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_MinMaxMerge_DATA_DATA(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		vx_int32 srcMinValue[AGO_MAX_PARAMS], srcMaxValue[AGO_MAX_PARAMS];
		vx_uint32 numDataPartitions = 0;
		for (vx_uint32 i = 3; i < node->paramCount; i++) {
			if (node->paramList[i]) {
				srcMinValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->min;
				srcMaxValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->max;
				numDataPartitions++;
			}
		}
		if (HafCpu_MinMaxMerge_DATA_DATA(&((ago_minmaxloc_data_t *)node->paramList[2]->buffer)->min,
			&((ago_minmaxloc_data_t *)node->paramList[2]->buffer)->max, numDataPartitions, srcMinValue, srcMaxValue))
		{
			status = VX_FAILURE;
		}
		else {
			// save the output values to output scalar values too
			node->paramList[0]->u.scalar.u.i = ((ago_minmaxloc_data_t *)node->paramList[2]->buffer)->min;
			node->paramList[1]->u.scalar.u.i = ((ago_minmaxloc_data_t *)node->paramList[2]->buffer)->max;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.scalar.type = node->paramList[0]->u.scalar.type;
		meta = &node->metaList[1];
		meta->data.u.scalar.type = node->paramList[1]->u.scalar.type;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_MinMaxLoc_DATA_U8DATA_Loc_None_Count_Min(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		vx_int32 srcMinValue[AGO_MAX_PARAMS], srcMaxValue[AGO_MAX_PARAMS];
		vx_uint32 numDataPartitions = 0;
		for (vx_uint32 i = 2; i < node->paramCount; i++) {
			if (node->paramList[i]) {
				srcMinValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->min;
				srcMaxValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->max;
				numDataPartitions++;
			}
		}
		AgoData * iImg = node->paramList[1];
		vx_int32 finalMinValue, finalMaxValue;
		if (HafCpu_MinMaxLoc_DATA_U8DATA_Loc_None_Count_Min(&node->paramList[0]->u.scalar.u.u, &finalMinValue, &finalMaxValue,
			numDataPartitions, srcMinValue, srcMaxValue, iImg->u.img.rect_valid.end_x - iImg->u.img.rect_valid.start_x, iImg->u.img.rect_valid.end_y - iImg->u.img.rect_valid.start_y,
			iImg->buffer + (iImg->u.img.rect_valid.start_y*iImg->u.img.stride_in_bytes) + iImg->u.img.rect_valid.start_x, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = VX_SUCCESS;
		// validate parameters
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!node->paramList[1]->u.img.width || !node->paramList[1]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_MinMaxLoc_DATA_U8DATA_Loc_None_Count_Max(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		vx_int32 srcMinValue[AGO_MAX_PARAMS], srcMaxValue[AGO_MAX_PARAMS];
		vx_uint32 numDataPartitions = 0;
		for (vx_uint32 i = 2; i < node->paramCount; i++) {
			if (node->paramList[i]) {
				srcMinValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->min;
				srcMaxValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->max;
				numDataPartitions++;
			}
		}
		AgoData * iImg = node->paramList[1];
		vx_int32 finalMinValue, finalMaxValue;
		if (HafCpu_MinMaxLoc_DATA_U8DATA_Loc_None_Count_Max(&node->paramList[0]->u.scalar.u.u, &finalMinValue, &finalMaxValue,
			numDataPartitions, srcMinValue, srcMaxValue, iImg->u.img.rect_valid.end_x - iImg->u.img.rect_valid.start_x, iImg->u.img.rect_valid.end_y - iImg->u.img.rect_valid.start_y, 
			iImg->buffer + (iImg->u.img.rect_valid.start_y*iImg->u.img.stride_in_bytes) + iImg->u.img.rect_valid.start_x, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = VX_SUCCESS;
		// validate parameters
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!node->paramList[1]->u.img.width || !node->paramList[1]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_MinMaxLoc_DATA_U8DATA_Loc_None_Count_MinMax(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		vx_int32 srcMinValue[AGO_MAX_PARAMS], srcMaxValue[AGO_MAX_PARAMS];
		vx_uint32 numDataPartitions = 0;
		for (vx_uint32 i = 3; i < node->paramCount; i++) {
			if (node->paramList[i]) {
				srcMinValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->min;
				srcMaxValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->max;
				numDataPartitions++;
			}
		}
		AgoData * iImg = node->paramList[2];
		vx_int32 finalMinValue, finalMaxValue;
		if (HafCpu_MinMaxLoc_DATA_U8DATA_Loc_None_Count_MinMax(&node->paramList[0]->u.scalar.u.u, &node->paramList[1]->u.scalar.u.u, &finalMinValue, &finalMaxValue,
			numDataPartitions, srcMinValue, srcMaxValue, iImg->u.img.rect_valid.end_x - iImg->u.img.rect_valid.start_x, iImg->u.img.rect_valid.end_y - iImg->u.img.rect_valid.start_y, 
			iImg->buffer + (iImg->u.img.rect_valid.start_y*iImg->u.img.stride_in_bytes) + iImg->u.img.rect_valid.start_x, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = VX_SUCCESS;
		// validate parameters
		if (node->paramList[2]->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!node->paramList[2]->u.img.width || !node->paramList[2]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
		meta = &node->metaList[1];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_MinMaxLoc_DATA_U8DATA_Loc_Min_Count_Min(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		vx_int32 srcMinValue[AGO_MAX_PARAMS], srcMaxValue[AGO_MAX_PARAMS];
		vx_uint32 numDataPartitions = 0;
		for (vx_uint32 i = 3; i < node->paramCount; i++) {
			if (node->paramList[i]) {
				srcMinValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->min;
				srcMaxValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->max;
				numDataPartitions++;
			}
		}
		AgoData * iMinLoc = node->paramList[0];
		AgoData * iMinCount = node->paramList[1];
		AgoData * iImg = node->paramList[2];
		vx_int32 finalMinValue, finalMaxValue;
		vx_uint32 minCount = 0;
		if (HafCpu_MinMaxLoc_DATA_U8DATA_Loc_Min_Count_Min(&minCount, (vx_uint32)iMinLoc->u.arr.capacity, (vx_coordinates2d_t *)iMinLoc->buffer, &finalMinValue, &finalMaxValue,
			numDataPartitions, srcMinValue, srcMaxValue, iImg->u.img.rect_valid.end_x - iImg->u.img.rect_valid.start_x, iImg->u.img.rect_valid.end_y - iImg->u.img.rect_valid.start_y, 
			iImg->buffer + (iImg->u.img.rect_valid.start_y*iImg->u.img.stride_in_bytes) + iImg->u.img.rect_valid.start_x, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
		else {
			iMinLoc->u.arr.numitems = min(minCount, (vx_uint32)iMinLoc->u.arr.capacity);
			if (iMinCount) iMinCount->u.scalar.u.u = minCount;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = VX_SUCCESS;
		// validate parameters
		AgoData * iImg = node->paramList[2];
		if (iImg->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!iImg->u.img.width || !iImg->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.arr.itemtype = VX_TYPE_COORDINATES2D;
		meta = &node->metaList[1];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_MinMaxLoc_DATA_U8DATA_Loc_Min_Count_MinMax(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		vx_int32 srcMinValue[AGO_MAX_PARAMS], srcMaxValue[AGO_MAX_PARAMS];
		vx_uint32 numDataPartitions = 0;
		for (vx_uint32 i = 4; i < node->paramCount; i++) {
			if (node->paramList[i]) {
				srcMinValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->min;
				srcMaxValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->max;
				numDataPartitions++;
			}
		}
		AgoData * iMinLoc = node->paramList[0];
		AgoData * iMinCount = node->paramList[1];
		AgoData * iMaxCount = node->paramList[2];
		AgoData * iImg = node->paramList[3];
		vx_int32 finalMinValue, finalMaxValue;
		vx_uint32 minCount = 0, maxCount = 0;
		if (HafCpu_MinMaxLoc_DATA_U8DATA_Loc_Min_Count_MinMax(&minCount, &maxCount, (vx_uint32)iMinLoc->u.arr.capacity, (vx_coordinates2d_t *)iMinLoc->buffer, &finalMinValue, &finalMaxValue,
			numDataPartitions, srcMinValue, srcMaxValue, iImg->u.img.rect_valid.end_x - iImg->u.img.rect_valid.start_x, iImg->u.img.rect_valid.end_y - iImg->u.img.rect_valid.start_y, 
			iImg->buffer + (iImg->u.img.rect_valid.start_y*iImg->u.img.stride_in_bytes) + iImg->u.img.rect_valid.start_x, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
		else {
			iMinLoc->u.arr.numitems = min(minCount, (vx_uint32)iMinLoc->u.arr.capacity);
			if (iMinCount) iMinCount->u.scalar.u.u = minCount;
			if (iMaxCount) iMaxCount->u.scalar.u.u = maxCount;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = VX_SUCCESS;
		// validate parameters
		AgoData * iImg = node->paramList[3];
		if (iImg->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!iImg->u.img.width || !iImg->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.arr.itemtype = VX_TYPE_COORDINATES2D;
		meta = &node->metaList[1];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
		meta = &node->metaList[2];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_MinMaxLoc_DATA_U8DATA_Loc_Max_Count_Max(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		vx_int32 srcMinValue[AGO_MAX_PARAMS], srcMaxValue[AGO_MAX_PARAMS];
		vx_uint32 numDataPartitions = 0;
		for (vx_uint32 i = 3; i < node->paramCount; i++) {
			if (node->paramList[i]) {
				srcMinValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->min;
				srcMaxValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->max;
				numDataPartitions++;
			}
		}
		AgoData * iMaxLoc = node->paramList[0];
		AgoData * iMaxCount = node->paramList[1];
		AgoData * iImg = node->paramList[2];
		vx_int32 finalMinValue, finalMaxValue;
		vx_uint32 maxCount = 0;
		if (HafCpu_MinMaxLoc_DATA_U8DATA_Loc_Max_Count_Max(&maxCount, (vx_uint32)iMaxLoc->u.arr.capacity, (vx_coordinates2d_t *)iMaxLoc->buffer, &finalMinValue, &finalMaxValue,
			numDataPartitions, srcMinValue, srcMaxValue, iImg->u.img.rect_valid.end_x - iImg->u.img.rect_valid.start_x, iImg->u.img.rect_valid.end_y - iImg->u.img.rect_valid.start_y,
			iImg->buffer + (iImg->u.img.rect_valid.start_y*iImg->u.img.stride_in_bytes) + iImg->u.img.rect_valid.start_x, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
		else {
			iMaxLoc->u.arr.numitems = min(maxCount, (vx_uint32)iMaxLoc->u.arr.capacity);
			if (iMaxCount) iMaxCount->u.scalar.u.u = maxCount;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = VX_SUCCESS;
		// validate parameters
		AgoData * iImg = node->paramList[2];
		if (iImg->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!iImg->u.img.width || !iImg->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.arr.itemtype = VX_TYPE_COORDINATES2D;
		meta = &node->metaList[1];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_MinMaxLoc_DATA_U8DATA_Loc_Max_Count_MinMax(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		vx_int32 srcMinValue[AGO_MAX_PARAMS], srcMaxValue[AGO_MAX_PARAMS];
		vx_uint32 numDataPartitions = 0;
		for (vx_uint32 i = 4; i < node->paramCount; i++) {
			if (node->paramList[i]) {
				srcMinValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->min;
				srcMaxValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->max;
				numDataPartitions++;
			}
		}
		AgoData * iMaxLoc = node->paramList[0];
		AgoData * iMinCount = node->paramList[1];
		AgoData * iMaxCount = node->paramList[2];
		AgoData * iImg = node->paramList[3];
		vx_int32 finalMinValue, finalMaxValue;
		vx_uint32 minCount = 0, maxCount = 0;
		if (HafCpu_MinMaxLoc_DATA_U8DATA_Loc_Max_Count_MinMax(&minCount, &maxCount, (vx_uint32)iMaxLoc->u.arr.capacity, (vx_coordinates2d_t *)iMaxLoc->buffer, &finalMinValue, &finalMaxValue,
			numDataPartitions, srcMinValue, srcMaxValue, iImg->u.img.rect_valid.end_x - iImg->u.img.rect_valid.start_x, iImg->u.img.rect_valid.end_y - iImg->u.img.rect_valid.start_y,
			iImg->buffer + (iImg->u.img.rect_valid.start_y*iImg->u.img.stride_in_bytes) + iImg->u.img.rect_valid.start_x, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
		else {
			iMaxLoc->u.arr.numitems = min(maxCount, (vx_uint32)iMaxLoc->u.arr.capacity);
			if (iMaxCount) iMaxCount->u.scalar.u.u = maxCount;
			if (iMinCount) iMinCount->u.scalar.u.u = minCount;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = VX_SUCCESS;
		// validate parameters
		AgoData * iImg = node->paramList[3];
		if (iImg->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!iImg->u.img.width || !iImg->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.arr.itemtype = VX_TYPE_COORDINATES2D;
		meta = &node->metaList[1];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
		meta = &node->metaList[2];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_MinMaxLoc_DATA_U8DATA_Loc_MinMax_Count_MinMax(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		vx_int32 srcMinValue[AGO_MAX_PARAMS], srcMaxValue[AGO_MAX_PARAMS];
		vx_uint32 numDataPartitions = 0;
		for (vx_uint32 i = 5; i < node->paramCount; i++) {
			if (node->paramList[i]) {
				srcMinValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->min;
				srcMaxValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->max;
				numDataPartitions++;
			}
		}
		AgoData * iMinLoc = node->paramList[0];
		AgoData * iMaxLoc = node->paramList[1];
		AgoData * iMinCount = node->paramList[2];
		AgoData * iMaxCount = node->paramList[3];
		AgoData * iImg = node->paramList[4];
		vx_int32 finalMinValue, finalMaxValue;
		vx_uint32 minCount = 0, maxCount = 0;
		if (HafCpu_MinMaxLoc_DATA_U8DATA_Loc_MinMax_Count_MinMax(&minCount, &maxCount, (vx_uint32)iMinLoc->u.arr.capacity, (vx_coordinates2d_t *)iMinLoc->buffer,
			(vx_uint32)iMaxLoc->u.arr.capacity, (vx_coordinates2d_t *)iMaxLoc->buffer, &finalMinValue, &finalMaxValue,
			numDataPartitions, srcMinValue, srcMaxValue, iImg->u.img.rect_valid.end_x - iImg->u.img.rect_valid.start_x, iImg->u.img.rect_valid.end_y - iImg->u.img.rect_valid.start_y,
			iImg->buffer + (iImg->u.img.rect_valid.start_y*iImg->u.img.stride_in_bytes) + iImg->u.img.rect_valid.start_x, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
		else {
			iMinLoc->u.arr.numitems = min(minCount, (vx_uint32)iMinLoc->u.arr.capacity);
			iMaxLoc->u.arr.numitems = min(maxCount, (vx_uint32)iMaxLoc->u.arr.capacity);
			if (iMinCount) iMinCount->u.scalar.u.u = minCount;
			if (iMaxCount) iMaxCount->u.scalar.u.u = maxCount;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = VX_SUCCESS;
		// validate parameters
		AgoData * iImg = node->paramList[4];
		if (iImg->u.img.format != VX_DF_IMAGE_U8)
			return VX_ERROR_INVALID_FORMAT;
		else if (!iImg->u.img.width || !iImg->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.arr.itemtype = VX_TYPE_COORDINATES2D;
		meta = &node->metaList[1];
		meta->data.u.arr.itemtype = VX_TYPE_COORDINATES2D;
		meta = &node->metaList[2];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
		meta = &node->metaList[3];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_MinMaxLoc_DATA_S16DATA_Loc_None_Count_Min(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		vx_int32 srcMinValue[AGO_MAX_PARAMS], srcMaxValue[AGO_MAX_PARAMS];
		vx_uint32 numDataPartitions = 0;
		for (vx_uint32 i = 2; i < node->paramCount; i++) {
			if (node->paramList[i]) {
				srcMinValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->min;
				srcMaxValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->max;
				numDataPartitions++;
			}
		}
		AgoData * iImg = node->paramList[1];
		vx_int32 finalMinValue, finalMaxValue;
		if (HafCpu_MinMaxLoc_DATA_S16DATA_Loc_None_Count_Min(&node->paramList[0]->u.scalar.u.u, &finalMinValue, &finalMaxValue,
			numDataPartitions, srcMinValue, srcMaxValue, iImg->u.img.rect_valid.end_x - iImg->u.img.rect_valid.start_x, iImg->u.img.rect_valid.end_y - iImg->u.img.rect_valid.start_y,
			(vx_int16 *)(iImg->buffer + (iImg->u.img.rect_valid.start_y*iImg->u.img.stride_in_bytes)) + iImg->u.img.rect_valid.start_x, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = VX_SUCCESS;
		// validate parameters
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_S16)
			return VX_ERROR_INVALID_FORMAT;
		else if (!node->paramList[1]->u.img.width || !node->paramList[1]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_MinMaxLoc_DATA_S16DATA_Loc_None_Count_Max(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		vx_int32 srcMinValue[AGO_MAX_PARAMS], srcMaxValue[AGO_MAX_PARAMS];
		vx_uint32 numDataPartitions = 0;
		for (vx_uint32 i = 2; i < node->paramCount; i++) {
			if (node->paramList[i]) {
				srcMinValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->min;
				srcMaxValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->max;
				numDataPartitions++;
			}
		}
		AgoData * iImg = node->paramList[1];
		vx_int32 finalMinValue, finalMaxValue;
		if (HafCpu_MinMaxLoc_DATA_S16DATA_Loc_None_Count_Max(&node->paramList[0]->u.scalar.u.u, &finalMinValue, &finalMaxValue,
			numDataPartitions, srcMinValue, srcMaxValue, iImg->u.img.rect_valid.end_x - iImg->u.img.rect_valid.start_x, iImg->u.img.rect_valid.end_y - iImg->u.img.rect_valid.start_y,
			(vx_int16 *)(iImg->buffer + (iImg->u.img.rect_valid.start_y*iImg->u.img.stride_in_bytes)) + iImg->u.img.rect_valid.start_x, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = VX_SUCCESS;
		// validate parameters
		if (node->paramList[1]->u.img.format != VX_DF_IMAGE_S16)
			return VX_ERROR_INVALID_FORMAT;
		else if (!node->paramList[1]->u.img.width || !node->paramList[1]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_MinMaxLoc_DATA_S16DATA_Loc_None_Count_MinMax(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		vx_int32 srcMinValue[AGO_MAX_PARAMS], srcMaxValue[AGO_MAX_PARAMS];
		vx_uint32 numDataPartitions = 0;
		for (vx_uint32 i = 3; i < node->paramCount; i++) {
			if (node->paramList[i]) {
				srcMinValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->min;
				srcMaxValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->max;
				numDataPartitions++;
			}
		}
		AgoData * iImg = node->paramList[2];
		vx_int32 finalMinValue, finalMaxValue;
		if (HafCpu_MinMaxLoc_DATA_S16DATA_Loc_None_Count_MinMax(&node->paramList[0]->u.scalar.u.u, &node->paramList[1]->u.scalar.u.u, &finalMinValue, &finalMaxValue,
			numDataPartitions, srcMinValue, srcMaxValue, iImg->u.img.rect_valid.end_x - iImg->u.img.rect_valid.start_x, iImg->u.img.rect_valid.end_y - iImg->u.img.rect_valid.start_y,
			(vx_int16 *)(iImg->buffer + (iImg->u.img.rect_valid.start_y*iImg->u.img.stride_in_bytes)) + iImg->u.img.rect_valid.start_x, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = VX_SUCCESS;
		// validate parameters
		if (node->paramList[2]->u.img.format != VX_DF_IMAGE_S16)
			return VX_ERROR_INVALID_FORMAT;
		else if (!node->paramList[2]->u.img.width || !node->paramList[2]->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
		meta = &node->metaList[1];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_MinMaxLoc_DATA_S16DATA_Loc_Min_Count_Min(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		vx_int32 srcMinValue[AGO_MAX_PARAMS], srcMaxValue[AGO_MAX_PARAMS];
		vx_uint32 numDataPartitions = 0;
		for (vx_uint32 i = 3; i < node->paramCount; i++) {
			if (node->paramList[i]) {
				srcMinValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->min;
				srcMaxValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->max;
				numDataPartitions++;
			}
		}
		AgoData * iMinLoc = node->paramList[0];
		AgoData * iMinCount = node->paramList[1];
		AgoData * iImg = node->paramList[2];
		vx_int32 finalMinValue, finalMaxValue;
		vx_uint32 minCount = 0;
		if (HafCpu_MinMaxLoc_DATA_S16DATA_Loc_Min_Count_Min(&minCount, (vx_uint32)iMinLoc->u.arr.capacity, (vx_coordinates2d_t *)iMinLoc->buffer, &finalMinValue, &finalMaxValue,
			numDataPartitions, srcMinValue, srcMaxValue, iImg->u.img.rect_valid.end_x - iImg->u.img.rect_valid.start_x, iImg->u.img.rect_valid.end_y - iImg->u.img.rect_valid.start_y,
			(vx_int16 *)(iImg->buffer + (iImg->u.img.rect_valid.start_y*iImg->u.img.stride_in_bytes)) + iImg->u.img.rect_valid.start_x, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
		else {
			iMinLoc->u.arr.numitems = min(minCount, (vx_uint32)iMinLoc->u.arr.capacity);
			if (iMinCount) iMinCount->u.scalar.u.u = minCount;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = VX_SUCCESS;
		// validate parameters
		AgoData * iImg = node->paramList[2];
		if (iImg->u.img.format != VX_DF_IMAGE_S16)
			return VX_ERROR_INVALID_FORMAT;
		else if (!iImg->u.img.width || !iImg->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.arr.itemtype = VX_TYPE_COORDINATES2D;
		meta = &node->metaList[1];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_MinMaxLoc_DATA_S16DATA_Loc_Min_Count_MinMax(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		vx_int32 srcMinValue[AGO_MAX_PARAMS], srcMaxValue[AGO_MAX_PARAMS];
		vx_uint32 numDataPartitions = 0;
		for (vx_uint32 i = 4; i < node->paramCount; i++) {
			if (node->paramList[i]) {
				srcMinValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->min;
				srcMaxValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->max;
				numDataPartitions++;
			}
		}
		AgoData * iMinLoc = node->paramList[0];
		AgoData * iMinCount = node->paramList[1];
		AgoData * iMaxCount = node->paramList[2];
		AgoData * iImg = node->paramList[3];
		vx_int32 finalMinValue, finalMaxValue;
		vx_uint32 minCount = 0, maxCount = 0;
		if (HafCpu_MinMaxLoc_DATA_S16DATA_Loc_Min_Count_MinMax(&minCount, &maxCount, (vx_uint32)iMinLoc->u.arr.capacity, (vx_coordinates2d_t *)iMinLoc->buffer, &finalMinValue, &finalMaxValue,
			numDataPartitions, srcMinValue, srcMaxValue, iImg->u.img.rect_valid.end_x - iImg->u.img.rect_valid.start_x, iImg->u.img.rect_valid.end_y - iImg->u.img.rect_valid.start_y,
			(vx_int16 *)(iImg->buffer + (iImg->u.img.rect_valid.start_y*iImg->u.img.stride_in_bytes)) + iImg->u.img.rect_valid.start_x, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
		else {
			iMinLoc->u.arr.numitems = min(minCount, (vx_uint32)iMinLoc->u.arr.capacity);
			if (iMinCount) iMinCount->u.scalar.u.u = minCount;
			if (iMaxCount) iMaxCount->u.scalar.u.u = maxCount;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = VX_SUCCESS;
		// validate parameters
		AgoData * iImg = node->paramList[3];
		if (iImg->u.img.format != VX_DF_IMAGE_S16)
			return VX_ERROR_INVALID_FORMAT;
		else if (!iImg->u.img.width || !iImg->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.arr.itemtype = VX_TYPE_COORDINATES2D;
		meta = &node->metaList[1];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
		meta = &node->metaList[2];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_MinMaxLoc_DATA_S16DATA_Loc_Max_Count_Max(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		vx_int32 srcMinValue[AGO_MAX_PARAMS], srcMaxValue[AGO_MAX_PARAMS];
		vx_uint32 numDataPartitions = 0;
		for (vx_uint32 i = 3; i < node->paramCount; i++) {
			if (node->paramList[i]) {
				srcMinValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->min;
				srcMaxValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->max;
				numDataPartitions++;
			}
		}
		AgoData * iMaxLoc = node->paramList[0];
		AgoData * iMaxCount = node->paramList[1];
		AgoData * iImg = node->paramList[2];
		vx_int32 finalMinValue, finalMaxValue;
		vx_uint32 maxCount = 0;
		if (HafCpu_MinMaxLoc_DATA_S16DATA_Loc_Max_Count_Max(&maxCount, (vx_uint32)iMaxLoc->u.arr.capacity, (vx_coordinates2d_t *)iMaxLoc->buffer, &finalMinValue, &finalMaxValue,
			numDataPartitions, srcMinValue, srcMaxValue, iImg->u.img.rect_valid.end_x - iImg->u.img.rect_valid.start_x, iImg->u.img.rect_valid.end_y - iImg->u.img.rect_valid.start_y,
			(vx_int16 *)(iImg->buffer + (iImg->u.img.rect_valid.start_y*iImg->u.img.stride_in_bytes)) + iImg->u.img.rect_valid.start_x, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
		else {
			iMaxLoc->u.arr.numitems = min(maxCount, (vx_uint32)iMaxLoc->u.arr.capacity);
			if (iMaxCount) iMaxCount->u.scalar.u.u = maxCount;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = VX_SUCCESS;
		// validate parameters
		AgoData * iImg = node->paramList[2];
		if (iImg->u.img.format != VX_DF_IMAGE_S16)
			return VX_ERROR_INVALID_FORMAT;
		else if (!iImg->u.img.width || !iImg->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.arr.itemtype = VX_TYPE_COORDINATES2D;
		meta = &node->metaList[1];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_MinMaxLoc_DATA_S16DATA_Loc_Max_Count_MinMax(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		vx_int32 srcMinValue[AGO_MAX_PARAMS], srcMaxValue[AGO_MAX_PARAMS];
		vx_uint32 numDataPartitions = 0;
		for (vx_uint32 i = 4; i < node->paramCount; i++) {
			if (node->paramList[i]) {
				srcMinValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->min;
				srcMaxValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->max;
				numDataPartitions++;
			}
		}
		AgoData * iMaxLoc = node->paramList[0];
		AgoData * iMinCount = node->paramList[1];
		AgoData * iMaxCount = node->paramList[2];
		AgoData * iImg = node->paramList[3];
		vx_int32 finalMinValue, finalMaxValue;
		vx_uint32 minCount = 0, maxCount = 0;
		if (HafCpu_MinMaxLoc_DATA_S16DATA_Loc_Max_Count_MinMax(&minCount, &maxCount, (vx_uint32)iMaxLoc->u.arr.capacity, (vx_coordinates2d_t *)iMaxLoc->buffer, &finalMinValue, &finalMaxValue,
			numDataPartitions, srcMinValue, srcMaxValue, iImg->u.img.rect_valid.end_x - iImg->u.img.rect_valid.start_x, iImg->u.img.rect_valid.end_y - iImg->u.img.rect_valid.start_y,
			(vx_int16 *)(iImg->buffer + (iImg->u.img.rect_valid.start_y*iImg->u.img.stride_in_bytes)) + iImg->u.img.rect_valid.start_x, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
		else {
			iMaxLoc->u.arr.numitems = min(maxCount, (vx_uint32)iMaxLoc->u.arr.capacity);
			if (iMinCount) iMinCount->u.scalar.u.u = minCount;
			if (iMaxCount) iMaxCount->u.scalar.u.u = maxCount;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = VX_SUCCESS;
		// validate parameters
		AgoData * iImg = node->paramList[3];
		if (iImg->u.img.format != VX_DF_IMAGE_S16)
			return VX_ERROR_INVALID_FORMAT;
		else if (!iImg->u.img.width || !iImg->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.arr.itemtype = VX_TYPE_COORDINATES2D;
		meta = &node->metaList[1];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
		meta = &node->metaList[2];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_MinMaxLoc_DATA_S16DATA_Loc_MinMax_Count_MinMax(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		vx_int32 srcMinValue[AGO_MAX_PARAMS], srcMaxValue[AGO_MAX_PARAMS];
		vx_uint32 numDataPartitions = 0;
		for (vx_uint32 i = 5; i < node->paramCount; i++) {
			if (node->paramList[i]) {
				srcMinValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->min;
				srcMaxValue[numDataPartitions] = ((ago_minmaxloc_data_t *)node->paramList[i]->buffer)->max;
				numDataPartitions++;
			}
		}
		AgoData * iMinLoc = node->paramList[0];
		AgoData * iMaxLoc = node->paramList[1];
		AgoData * iMinCount = node->paramList[2];
		AgoData * iMaxCount = node->paramList[3];
		AgoData * iImg = node->paramList[4];
		vx_int32 finalMinValue, finalMaxValue;
		vx_uint32 minCount = 0, maxCount = 0;
		if (HafCpu_MinMaxLoc_DATA_S16DATA_Loc_MinMax_Count_MinMax(&minCount, &maxCount, (vx_uint32)iMinLoc->u.arr.capacity, (vx_coordinates2d_t *)iMinLoc->buffer,
			(vx_uint32)iMaxLoc->u.arr.capacity, (vx_coordinates2d_t *)iMaxLoc->buffer, &finalMinValue, &finalMaxValue,
			numDataPartitions, srcMinValue, srcMaxValue, iImg->u.img.rect_valid.end_x - iImg->u.img.rect_valid.start_x, iImg->u.img.rect_valid.end_y - iImg->u.img.rect_valid.start_y,
			(vx_int16 *)(iImg->buffer + (iImg->u.img.rect_valid.start_y*iImg->u.img.stride_in_bytes)) + iImg->u.img.rect_valid.start_x, iImg->u.img.stride_in_bytes))
		{
			status = VX_FAILURE;
		}
		else {
			iMinLoc->u.arr.numitems = min(minCount, (vx_uint32)iMinLoc->u.arr.capacity);
			iMaxLoc->u.arr.numitems = min(maxCount, (vx_uint32)iMaxLoc->u.arr.capacity);
			if (iMinCount) iMinCount->u.scalar.u.u = minCount;
			if (iMaxCount) iMaxCount->u.scalar.u.u = maxCount;
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		status = VX_SUCCESS;
		// validate parameters
		AgoData * iImg = node->paramList[4];
		if (iImg->u.img.format != VX_DF_IMAGE_S16)
			return VX_ERROR_INVALID_FORMAT;
		else if (!iImg->u.img.width || !iImg->u.img.height)
			return VX_ERROR_INVALID_DIMENSION;
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.arr.itemtype = VX_TYPE_COORDINATES2D;
		meta = &node->metaList[1];
		meta->data.u.arr.itemtype = VX_TYPE_COORDINATES2D;
		meta = &node->metaList[2];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
		meta = &node->metaList[3];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
    else if (cmd == ago_kernel_cmd_query_target_support) {
        node->target_support_flags = 0
                    | AGO_KERNEL_FLAG_DEVICE_CPU
                    ;
        status = VX_SUCCESS;
    }
	return status;
}

int agoKernel_MinMaxLocMerge_DATA_DATA(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		status = VX_SUCCESS;
		vx_uint32 numDataPartitions = 0;
		vx_uint32 partLocCount[AGO_MAX_PARAMS];
		vx_coordinates2d_t * partLocList[AGO_MAX_PARAMS];
		for (vx_uint32 i = 2; i < node->paramCount; i++) {
			if (node->paramList[i] && node->paramList[i]->u.arr.numitems) {
				partLocCount[numDataPartitions] = (vx_uint32) node->paramList[i]->u.arr.numitems;
				partLocList[numDataPartitions] = (vx_coordinates2d_t *)node->paramList[i]->buffer;
				numDataPartitions++;
			}
		}
		vx_uint32 countMinMaxLoc = 0;
		if (HafCpu_MinMaxLocMerge_DATA_DATA(&node->paramList[0]->u.scalar.u.u, (vx_uint32)node->paramList[1]->u.arr.capacity, (vx_coordinates2d_t *)node->paramList[1]->buffer,
			numDataPartitions, partLocCount, partLocList))
		{
			status = VX_FAILURE;
		}
		else {
			node->paramList[1]->u.arr.numitems = min(node->paramList[0]->u.scalar.u.u, (vx_uint32)node->paramList[1]->u.arr.capacity);
		}
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate inputs
		for (vx_uint32 i = 2; i < node->paramCount; i++) {
			if (node->paramList[i]) {
				if (node->paramList[i]->u.arr.itemtype != VX_TYPE_COORDINATES2D)
					return VX_ERROR_INVALID_TYPE;
			}
		}
		// set output info
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.u.scalar.type = VX_TYPE_UINT32;
		meta = &node->metaList[1];
		meta->data.u.arr.itemtype = VX_TYPE_COORDINATES2D;
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0
			| AGO_KERNEL_FLAG_DEVICE_CPU
			;
		status = VX_SUCCESS;
	}
	return status;
}

int agoKernel_Copy_DATA_DATA(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		// TBD: not implemented yet
		status = VX_ERROR_NOT_SUPPORTED;
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// validate parameters
		if (node->paramList[0]->ref.type != node->paramList[1]->ref.type)
			return VX_ERROR_INVALID_PARAMETERS;
		// doesn't support host access buffers
		if (node->paramList[0]->import_type != VX_MEMORY_TYPE_NONE || node->paramList[1]->import_type != VX_MEMORY_TYPE_NONE)
			return VX_ERROR_NOT_SUPPORTED;
		// doesn't support ROIs
		if ((node->paramList[0]->ref.type == VX_TYPE_IMAGE  && node->paramList[0]->u.img.roiMasterImage) ||
		    (node->paramList[1]->ref.type == VX_TYPE_IMAGE  && node->paramList[1]->u.img.roiMasterImage) ||
		    (node->paramList[0]->ref.type == VX_TYPE_TENSOR && node->paramList[0]->u.tensor.roiMaster) ||
		    (node->paramList[1]->ref.type == VX_TYPE_TENSOR && node->paramList[1]->u.tensor.roiMaster))
			return VX_ERROR_NOT_SUPPORTED;
		// set meta must be same as input
		vx_meta_format meta;
		meta = &node->metaList[0];
		meta->data.ref.type = node->paramList[1]->ref.type;
		memcpy(&meta->data.u, &node->paramList[1]->u, sizeof(meta->data.u));
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		// TBD: not implemented yet
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		size_t work_group_size = 256;
		size_t num_work_items = node->paramList[0]->size / 4;
		char code[1024];
		sprintf(code,
			"__kernel __attribute__((reqd_work_group_size(%zd, 1, 1)))\n"
			"void %s(__global char * dst_buf, uint dst_offset, uint4 dst_stride, __global char * src_buf, uint src_offset, uint4 src_stride)\n"
			"{\n"
			"    uint id = get_global_id(0);\n"
			"    if(id < %zd) ((__global float *)(dst_buf + dst_offset))[id] =  ((__global float *)(src_buf + src_offset))[id];\n"
			"}\n", work_group_size, NODE_OPENCL_KERNEL_NAME, num_work_items);
		node->opencl_code = code;
		// use completely separate kernel
		node->opencl_type = NODE_OPENCL_TYPE_FULL_KERNEL;
		node->opencl_work_dim = 3;
		node->opencl_global_work[0] = (num_work_items + work_group_size - 1) & ~(work_group_size - 1);
		node->opencl_global_work[1] = 1;
		node->opencl_global_work[2] = 1;
		node->opencl_local_work[0] = work_group_size;
		node->opencl_local_work[1] = 1;
		node->opencl_local_work[2] = 1;
		status = VX_SUCCESS;
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0;
#if ENABLE_OPENCL
		if (node->paramList[0]->ref.type == VX_TYPE_TENSOR)
			node->target_support_flags |= AGO_KERNEL_FLAG_DEVICE_GPU | AGO_KERNEL_FLAG_GPU_INTEG_FULL;
#endif
        status = VX_SUCCESS;
	}
	return status;
}

int agoKernel_Select_DATA_DATA_DATA(AgoNode * node, AgoKernelCommand cmd)
{
	vx_status status = AGO_ERROR_KERNEL_NOT_IMPLEMENTED;
	if (cmd == ago_kernel_cmd_execute) {
		// TBD: not implemented yet
		status = VX_ERROR_NOT_SUPPORTED;
	}
	else if (cmd == ago_kernel_cmd_validate) {
		// TBD: not implemented yet
		status = VX_ERROR_NOT_SUPPORTED;
	}
	else if (cmd == ago_kernel_cmd_initialize || cmd == ago_kernel_cmd_shutdown) {
		status = VX_SUCCESS;
	}
	else if (cmd == ago_kernel_cmd_valid_rect_callback) {
		// TBD: not implemented yet
	}
#if ENABLE_OPENCL
	else if (cmd == ago_kernel_cmd_opencl_codegen) {
		// TBD: not implemented yet
		status = VX_ERROR_NOT_SUPPORTED;
	}
#endif
	else if (cmd == ago_kernel_cmd_query_target_support) {
		node->target_support_flags = 0;
		status = VX_SUCCESS;
	}
	return status;
}

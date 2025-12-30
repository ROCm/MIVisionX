/*
Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef _VX_EXT_RPP_H_
#define _VX_EXT_RPP_H_

#include "VX/vx.h"
#include "VX/vx_compatibility.h"

/*!
 * \file
 * \brief The AMD OpenVX RPP Nodes Extension Library.
 *

 * \defgroup group_amd_rpp Extension: AMD RPP Extension API
 * \brief AMD OpenVX RPP Nodes Extension to use as the low-level library for rocAL.
 */

#ifndef dimof
/*! \def dimof(x)
 *  \brief A macro to get the number of elements in an array.
 *  \param [in] x The array whose size is to be determined.
 *  \return The number of elements in the array.
 */
#define dimof(x) (sizeof(x) / sizeof(x[0]))
#endif

#ifndef SHARED_PUBLIC
#if _WIN32
#define SHARED_PUBLIC __declspec(dllexport)
#else
/*! \def SHARED_PUBLIC
 *  \brief A macro to specify public visibility for shared library symbols.
 */
#define SHARED_PUBLIC __attribute__((visibility("default")))
#endif
#endif

/*! \brief Creates a node in a graph using a predefined kernel structure.
 *  \param [in] graph The handle to the graph.
 *  \param [in] kernelenum The enum value representing the kernel to be used.
 *  \param [in] params An array of parameter references for the kernel.
 *  \param [in] num The number of parameters in the params array.
 *  \return A handle to the created node.
 */
vx_node vxCreateNodeByStructure(vx_graph graph, vx_enum kernelenum, vx_reference params[], vx_uint32 num);

#ifdef __cplusplus
extern "C"
{
#endif

	/*!***********************************************************************************************************
								 RPP VX_API_ENTRY C Function NODE
	*************************************************************************************************************/
	// Tensor Augmentations
	/*! \brief [Graph] Creates a Brightness function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pAlpha The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the alpha data.
	 * \param [in] pBeta The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the beta data.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppBrightness(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pAlpha, vx_array pBeta, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);

	/*! \brief [Graph] Creates a Copy function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor data.
	 * \param [out] pDst The output tensor data.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppCopy(vx_graph graph, vx_tensor pSrc, vx_tensor pDst);

	/*! \brief [Graph] Creates a CropMirrorNormalize function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pMultiplier The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the multiplier data.
	 * \param [in] pOffset The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the offset data.
	 * \param [in] pMirror The input array in <tt>\ref VX_TYPE_INT32</tt> format containing the flip data.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppCropMirrorNormalize(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pMultiplier, vx_array pOffset, vx_array pMirror, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);

	/*! \brief [Graph] Creates a Nop function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor data.
	 * \param [out] pDst The output tensor data.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppNop(vx_graph graph, vx_tensor pSrc, vx_tensor pDst);

	/*! \brief [Graph] Creates a Resize function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pDstWidth The input array in <tt>\ref VX_TYPE_UINT32</tt> format containing the output width data.
	 * \param [in] pDstHeight The input array in <tt>\ref VX_TYPE_UINT32</tt> format containing the output height data.
	 * \param [in] interpolationType The resize interpolation type in <tt>\ref VX_TYPE_INT32</tt> format containing the type of interpolation.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppResize(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pDstWidth, vx_array pDstHeight, vx_scalar interpolationType, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);

	/*!
	 * \brief [Graph] Creates a SequenceRearrange function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor <tt>\ref VX_TYPE_UINT8</tt> format data.
	 * \param [out] pDst The output tensor <tt>\ref VX_TYPE_UINT8</tt> format data.
	 * \param [in] pNewOrder The rearrange order in <tt>\ref VX_TYPE_UINT32</tt> containing the order in which frames are copied.
	 * \param [in] layout The layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input and output tensor.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppSequenceRearrange(vx_graph graph, vx_tensor pSrc, vx_tensor pDst, vx_array pNewOrder, vx_scalar layout);

	/*! \brief [Graph] Creates a Blend function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc1 The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrc2 The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pShift The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the shift data.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppBlend(vx_graph graph, vx_tensor pSrc1, vx_tensor pSrc2, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pShift, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a Blur function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int<tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> format data.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppBlur(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a ColorTemperature function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pAdjustValue The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the adjustment value data.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppColorTemperature(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pAdjustValue, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a ColorTwist function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pAlpha The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the alpha data.
	 * \param [in] pBeta The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the beta data.
	 * \param [in] pHue The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the hue data.
	 * \param [in] pSat The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the saturation data.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppColorTwist(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pAlpha, vx_array pBeta, vx_array pHue, vx_array pSat, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a Contrast function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pContrastFactor The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the contrast factor data.
	 * \param [in] pContrastCenter The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the contrast center data.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppContrast(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pContrastFactor, vx_array pContrastCenter, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a Crop function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppCrop(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a Exposure function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pExposureFactor The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the exposure factor data.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppExposure(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pExposureFactor, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a FishEye function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The tensor in <tt>\ref VX_TYPE_UINT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> format data.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppFishEye(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a Flip function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pHflag The input array in <tt>\ref VX_TYPE_UINT32</tt> format containing the horizontal flag data.
	 * \param [in] pVflag The input array in <tt>\ref VX_TYPE_UINT32</tt> format containing the vertical flag data.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppFlip(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pHflag, vx_array pVflag, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a Fog function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> format data.
	 * \param [in] pIntensityFactor The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the intensity factor values for fog calculation.
	 * \param [in] pGrayFactor The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the gray factor values to introduce grayness in the image for fog calculation.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppFog(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pIntensityFactor, vx_array pGrayFactor, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a GammaCorrection function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pGamma The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the gamma data.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppGammaCorrection(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pGamma, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a Glitch function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pXoffsetR The input array in <tt>\ref VX_TYPE_UINT32</tt> format containing the x offset for r-channel data.
	 * \param [in] pYoffsetR The input array in <tt>\ref VX_TYPE_UINT32</tt> format containing the y offset for r-channel data.
	 * \param [in] pXoffsetG The input array in <tt>\ref VX_TYPE_UINT32</tt> format containing the x offset for g-channel data.
	 * \param [in] pYoffsetG The input array in <tt>\ref VX_TYPE_UINT32</tt> format containing the y offset for g-channel data.
	 * \param [in] pXoffsetB The input array in <tt>\ref VX_TYPE_UINT32</tt> format containing the x offset for b-channel data.
	 * \param [in] pYoffsetB The input array in <tt>\ref VX_TYPE_UINT32</tt> format containing the y offset for b-channel data.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppGlitch(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pXoffsetR, vx_array pYoffsetR, vx_array pXoffsetG, vx_array pYoffsetG, vx_array pXoffsetB, vx_array pYoffsetB, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a Hue function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pHueShift The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the hue shift data.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppHue(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pHueShift, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a Jitter function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pKernelSize The input array in <tt>\ref VX_TYPE_UINT32</tt> format containing the kernel size data.
	 * \param [in] seed The input scalar in <tt>\ref VX_TYPE_UINT32</tt> contains the seed value.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppJitter(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pKernelSize, vx_scalar seed, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a LensCorrection function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> format data.
	 * \param [in] pCameraMatrix The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing camera intrinsic parameters required to compute lens corrected image.
	 * \param [in] pDistortionCoeffs The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the distortion coefficients required to compute lens corrected image.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppLensCorrection(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pCameraMatrix, vx_array pDistortionCoeffs, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a Noise function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pNoiseProb The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the noise probability data.
	 * \param [in] pSaltProb The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the salt probability data.
	 * \param [in] pSaltValue The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the salt value data.
	 * \param [in] pPepperValue The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the pepper value data.
	 * \param [in] seed The input scalar in <tt>\ref VX_TYPE_UINT32</tt> contains the seed value.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppNoise(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pNoiseProb, vx_array pSaltProb, vx_array pSaltValue, vx_array pPepperValue, vx_scalar seed, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a Noise function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] rainPercentage The input value in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the percentage of the rain effect to be applied.
	 * \param [in] rainWidth The input value in <tt>\ref VX_TYPE_UINT32</tt> format containing the width of the rain drops in pixels.
	 * \param [in] rainHeight The input value in <tt>\ref VX_TYPE_UINT32</tt> format containing the height of the rain drops in pixels.
	 * \param [in] rainSlantAngle The input value in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the slant angle of the rain drops (positive value for right slant, negative for left slant)
	 * \param [in] pRainTransperancy The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the rain transparency data.
	 * \param [in] seed The input scalar in <tt>\ref VX_TYPE_UINT32</tt> contains the seed value.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppRain(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_scalar rainPercentage, vx_scalar rainWidth, vx_scalar rainHeight, vx_scalar rainSlantAngle, vx_array pRainTransperancy, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a ResizeCrop function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pDstWidth The input array in <tt>\ref VX_TYPE_UINT32</tt> format containing the output width data.
	 * \param [in] pDstHeight The input array in <tt>\ref VX_TYPE_UINT32</tt> format containing the output height data.
	 * \param [in] interpolationType The resize interpolation type in <tt>\ref VX_TYPE_INT32</tt> format containing the type of interpolation.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppResizeCrop(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pDstWidth, vx_array pDstHeight, vx_scalar interpolationType, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a ResizeCropMirror function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pDstWidth The input array in <tt>\ref VX_TYPE_UINT32</tt> format containing the output width data.
	 * \param [in] pDstHeight The input array in <tt>\ref VX_TYPE_UINT32</tt> format containing the output height data.
	 * \param [in] pMirror The input array in <tt>\ref VX_TYPE_INT32</tt> format containing the mirror data.
	 * \param [in] interpolationType The resize interpolation type in <tt>\ref VX_TYPE_INT32</tt> format containing the type of interpolation.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppResizeCropMirror(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pDstWidth, vx_array pDstHeight, vx_array pMirror, vx_scalar interpolationType, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a ResizeMirrorNormalize function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pDstWidth The input array in <tt>\ref VX_TYPE_UINT32</tt> format containing the output width data.
	 * \param [in] pDstHeight The input array in <tt>\ref VX_TYPE_UINT32</tt> format containing the output height data.
	 * \param [in] interpolationType The resize interpolation type in <tt>\ref VX_TYPE_INT32</tt> format containing the type of interpolation.
	 * \param [in] pMean The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the mean data.
	 * \param [in] pStdDev The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the std-dev data.
	 * \param [in] pMirror The input array in <tt>\ref VX_TYPE_INT32</tt> format containing the mirror data.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppResizeMirrorNormalize(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst,vx_array pDstWidth, vx_array pDstHeight, vx_scalar interpolationType, vx_array pMean, vx_array pStdDev, vx_array pMirror, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a Rotate function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pAngle The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the angle data.
	 * \param [in] interpolationType The resize interpolation type in <tt>\ref VX_TYPE_INT32</tt> format containing the type of interpolation.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppRotate(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pAngle, vx_scalar interpolationType, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a Saturation function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSaturationFactor The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the saturation factor data.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppSaturation(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pSaturationFactor, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a Snow function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> format data.
	 * \param [in] pBrightnessCoefficient The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the brightness coefficient (per image). Valid range: (1, 4].
	 * \param [in] pSnowThreshold The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the snow threshold (per image). Valid range: (0, 1].
	 * \param [in] pDarkMode The input array in <tt>\ref VX_TYPE_INT32</tt> format containing the dark mode enable/disable flag (per image). Valid values: 0/1.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppSnow(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pBrightnessCoefficient, vx_array pSnowThreshold, vx_array pDarkMode, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a Pixelate function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pixelationPercentage The input value in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the variable controling how much pixelation is applied to images.(pixelationPercentage value ranges from 0 to 100).
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppPixelate(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_scalar pixelationPercentage, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);
	
	/*! \brief [Graph] Creates a Vignette function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pStdDev The input array in <tt>VX_TYPE_FLOAT32</tt> format containing the standard deviation data.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppVignette(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pStdDev, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);

	/*! \brief [Graph] Creates a Warp-Affine function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh/ltrb format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> format data.
	 * \param [in] pAffineArray The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the affine transformation data.
	 * \param [in] interpolationType The resize interpolation type in <tt>\ref VX_TYPE_INT32</tt> format containing the type of interpolation.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The type of roi <tt>\ref VX_TYPE_INT32</tt> denotes whether source roi is of XYWH/LTRB type.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppWarpAffine(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pAffineArray, vx_scalar interpolationType, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);


	/*!
	 * \brief [Graph] Creates a Tensor SequenceRearrange function node.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8<tt> format data.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8<tt> format data.
	 * \param [in] pNewOrder The rearrange order in <tt>\ref VX_TYPE_UINT32<tt> containing the order in which frames are copied.
	 * \param [in] layout The layout in <tt>\ref VX_TYPE_INT32<tt> denotes the layout of input and output tensor.
	 * \return <tt> vx_node</tt>.
	 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
	 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppSequenceRearrange(vx_graph graph, vx_tensor pSrc, vx_tensor pDst, vx_array pNewOrder, vx_scalar layout);

	/*! \brief [Graph] Applies preemphasis filter to the input tensor.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_FLOAT32</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input in xywh (w- samples, h - channels) format.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_FLOAT32</tt> format data.
	 * \param [in] pPreemphCoeff The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the preEmphasis co-efficient.
	 * \param [in] borderType The type of border <tt>\ref VX_TYPE_INT32</tt> which can be "zero", "clamp", "reflect".
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppPreemphasisFilter(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pPreemphCoeff, vx_scalar borderType);

	/*! \brief [Graph] Produces a spectrogram from a 1D signal.
	* \ingroup group_amd_rpp
	* \param [in] graph The handle to the graph.
	* \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_FLOAT32</tt> format data.
	* \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int<tt> containing the roi values for the input in xywh/ltrb format.
	* \param [out] pDst The output tensor (begin) in <tt>\ref VX_TYPE_FLOAT32</tt> format data.
	* \param [in] pDstRoi The input tensor of batch size in <tt>unsigned int<tt> containing the roi values for the output tensor in xywh/ltrb format.
	* \param [in] windowFunction The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the samples of the window function that will be multiplied to each extracted window when calculating the STFT.
	* \param [in] centerWindow The input scalar in <tt>\ref VX_TYPE_BOOL</tt> format indicates whether extracted windows should be padded so that the window function is centered at multiples of window_step.
	* \param [in] reflectPadding The input scalar in <tt>\ref VX_TYPE_BOOL</tt> format indicates the padding policy when sampling outside the bounds of the signal.
	* \param [in] spectrogramLayout The input scalar in <tt>\ref VX_TYPE_INT32</tt> format containing the Output spectrogram layout.
	* \param [in] power The input scalar in <tt>\ref VX_TYPE_INT32</tt> format containing the exponent of the magnitude of the spectrum.
	* \param [in] nfft The input scalar in <tt>\ref VX_TYPE_INT32</tt> format containing the size of the FFT.
	* \param [in] windowLength The input scalar in <tt>\ref VX_TYPE_INT32</tt> format containing Window size in number of samples.
	* \param [in] windowStep The input array in <tt>\ref VX_TYPE_INT32</tt> format containing the step between the STFT windows in number of samples.
	* \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	*/
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppSpectrogram(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_tensor pDstRoi, vx_array windowFunction, vx_scalar centerWindow, vx_scalar reflectPadding, vx_scalar spectrogramLayout, vx_scalar power, vx_scalar nfft, vx_scalar windowLength, vx_scalar windowStep);

	/*! \brief [Graph] Applies downmixing to the input tensor.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_FLOAT32</tt> format data.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_FLOAT32</tt> format data.
     * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int<tt> containing the roi values for the input.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppDownmix(vx_graph graph, vx_tensor pSrc, vx_tensor pDst, vx_tensor srcRoi);

	/*! \brief [Graph] Applies to_decibels augmentation to the input tensor.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_FLOAT32</tt> format data.
	 * \param[in] pSrcRoi The input tensor of batch size in <tt>unsigned int<tt> containing the roi values for the input.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_FLOAT32</tt> format data.
	 * \param[in] cutOffDB The input scalar in <tt>\ref VX_TYPE_FLOAT32</tt> format containing  minimum or cut-off ratio in dB
	 * \param[in] multiplier The input scalar in <tt>\ref VX_TYPE_FLOAT32</tt> format containing factor by which the logarithm is multiplied
	 * \param[in] referenceMagnitude The input scalar in <tt>\ref VX_TYPE_FLOAT32</tt> format containing Reference magnitude which if not provided uses maximum value of input as reference
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppToDecibels(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_scalar cutOffDB, vx_scalar multiplier, vx_scalar referenceMagnitude, vx_scalar inputLayout, vx_scalar outputLayout);

	/*! \brief [Graph] Applies resample augmentation to the input tensor.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_FLOAT32</tt> format data.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_FLOAT32</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int<tt> containing the roi values for the input.
	 * \param [in] pDstRoi The input tensor of batch size in <tt>unsigned int<tt> containing the roi values for the output.
	 * \param [in] pInRateTensor The input array in <tt>\ref VX_TYPE_FLOAT32<tt> format containing the input sample rate data.
	 * \param [in] pOutRateTensor The input tensor in <tt>\ref VX_TYPE_FLOAT32<tt> format containing the output sample rate data.
	 * \param [in] quality The resampling is achieved by applying a sinc filter with Hann window with an extent controlled by the quality argument.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppResample(vx_graph graph, vx_tensor pSrc, vx_tensor pDst, vx_tensor pSrcRoi, vx_tensor pDstRoi, vx_array pInRateTensor, vx_tensor pOutRateTensor, vx_scalar quality);

	/*! \brief [Graph] Multiples a tensor and a scalar and returns the output.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_FLOAT32<tt> format data.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_FLOAT32<tt> format data.
	 * \param [in] scalarValue The input scalar in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the value used to multiply the input tensor
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppTensorMulScalar(vx_graph graph, vx_tensor pSrc, vx_tensor pDst, vx_scalar scalarValue);

	/*! \brief [Graph] Adds two tensors and returns the output.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc1 The input tensor1 in <tt>\ref VX_TYPE_FLOAT32<tt> format data.
	 * \param [in] pSrc2 The input tensor2 in <tt>\ref VX_TYPE_FLOAT32<tt> format data.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_FLOAT32<tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int<tt> containing the roi values for the input1.
	 * \param [in] pDstRoi The input tensor of batch size in <tt>unsigned int<tt> containing the roi values for the input2.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppTensorAddTensor(vx_graph graph, vx_tensor pSrc1, vx_tensor pSrc2, vx_tensor pDst, vx_tensor pSrcRoi, vx_tensor pDstRoi);

	/*! \brief [Graph] Performs leading and trailing silence detection to the input tensor.
	* \ingroup group_amd_rpp
	* \param [in] graph The handle to the graph.
	* \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_FLOAT32</tt> format data.
	* \param [in] pSrcRoi The input tensor in <tt>unsigned int<tt> containing the roi values for each input in the format <begin_dim1, begin_dim2 .., length_dim1, length_dim2> format for each dimension.
	* \param [out] pBegin The output tensor containing begin values of the non-silent region of the audio-data in <tt>\ref VX_TYPE_INT32</tt> format data.
	* \param [out] pLength The output tensor containing length values of the non-silent region of the audio-data in <tt>\ref VX_TYPE_INT32</tt> format data.
	* \param [in] cutOffDB The input scalar in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the threshold, in dB, below which the signal is considered silent.
	* \param [in] referencePower The input scalar in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the reference power used for converting the signal to dB.
	* \param [in] windowLength The input scalar in <tt>\ref VX_TYPE_INT32</tt> format containing the size of the sliding window.
	* \param [in] resetInterval The input scalar in <tt>\ref VX_TYPE_INT32</tt> format containing the frequency at which the moving mean average is recalculated to mitigate precision loss.
	* \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	*/
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppNonSilentRegionDetection(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst1, vx_tensor pDst2, vx_scalar cutOffDB, vx_scalar referencePower, vx_scalar windowLength, vx_scalar resetInterval);

	/*! \brief [Graph] Slice's the input tensor using anchors and shapes values
	* \ingroup group_amd_rpp
	* \param [in] graph The handle to the graph.
	* \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> format data.
	* \param [in] pSrcRoi The input tensor in <tt>unsigned int<tt> format containing the roi values for the input format <begin_dim1, begin_dim2 .., length_dim1, length_dim2> format for each dimension.
	* \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> format data.
	* \param [in] pDstRoi The input tensor in <tt>unsigned int<tt> format containing the roi values for the output tensor format <begin_dim1, begin_dim2 .., length_dim1, length_dim2> format for each dimension.
	* \param [in] pAnchor The input array in <tt>\ref VX_TYPE_INT32</tt> format containing the absolute coordinates for the starting point of the slice.
	* \param [in] pShape The input array in <tt>\ref VX_TYPE_INT32</tt> format containing the absolute coordinates for the dimensions of the slice
	* \param [in] pFillValue The input array in <tt>\ref VX_TYPE_INT32</tt> format which determines the values to be padded and is only relevant if policy is set to “pad”.
	* \param [in] policy The input scalar in <tt>\ref VX_TYPE_INT32</tt> format which determines the policy when slicing the out of bounds area of the input. The values can be "error", "pad", "trim_to_shape"
	* \param [in] inputLayout The input scalar in <tt>\ref VX_TYPE_INT32</tt> format containing the layout of the input tensor.
	* \param [in] roiType The input scalar in <tt>\ref VX_TYPE_INT32</tt> format containing the roi type which can be ltrb/xywh.
	* \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	*/
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppSlice(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_tensor pDstRoi, vx_tensor pAnchor, vx_tensor pShape, vx_array pFillValue, vx_scalar policy, vx_scalar inputLayout, vx_scalar roiType);

	/*! \brief [Graph] Applies mean-stddev normalization to the input tensor.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_FLOAT32</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int<tt> containing the roi values for the input.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_FLOAT32</tt> format data.
	 * \param [in] pDstRoi The input tensor of batch size in <tt>unsigned int<tt> containing the roi values for the output.
	 * \param [in] axisMask The input array in <tt>\ref VX_TYPE_INT32</tt> format containing axis along which normalization needs to be done
	 * \param [in] pMean The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing values to be subtracted from input
	 * \param [in] pStddev The input array in <tt>\ref VX_TYPE_FLOAT32</tt> format containing standard deviation values to scale the input
	 * \param [in] computeMeanAndStddev The input scalar in <tt>\ref VX_TYPE_UINT8</tt> format containing flag to represent internal computation of mean and std dev
	 * \param [in] scale The input scalar in <tt>\ref VX_TYPE_FLOAT32</tt> format containing value to be multiplied with data after subtracting from mean
	 * \param [in] shift The input scalar in <tt>\ref VX_TYPE_FLOAT32</tt> format containing value to be added finally
	 * \param [in] inputLayout The input scalar in <tt>\ref VX_TYPE_INT32</tt> format containing the layout of the input tensor.
	 * \param [in] roiType The input scalar in <tt>\ref VX_TYPE_INT32</tt> format containing the roi type which can be ltrb/xywh.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppNormalize(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_tensor pDstRoi, vx_scalar axis_mask, vx_array pMean, vx_array pStddev, vx_scalar computeMeanAndStddev, vx_scalar scale, vx_scalar shift, vx_scalar inputLayout, vx_scalar roiType);

	/*! \brief [Graph] Produces a mel-spectrogram from spectrogram on applying a bank of triangular filters
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_FLOAT32</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int<tt> containing the roi values for the input in xywh (w- samples, h - channels) format.
	 * \param [out] pDst The output tensor (begin) in <tt>\ref VX_TYPE_FLOAT32</tt> format data.
	 * \param [in] pDstRoi The input tensor of batch size in <tt>unsigned int<tt> containing the roi values for the input in xywh (w- samples, h - channels) format.
	 * \param [in] freqHigh The input scalar in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the maximum frequency.
	 * \param [in] freqLow The input scalar in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the minimum frequency.
	 * \param [in] melFormula The input scalar in <tt>\ref VX_TYPE_INT32</tt> format indicates the formula used to convert frequencies from hertz to mel and vice versa.
	 * \param [in] nfilter The input scalar in <tt>\ref VX_TYPE_INT32</tt> format containing the number of mel filters.
	 * \param [in] normalize The input scalar in <tt>\ref VX_TYPE_BOOL</tt> format to determine weather to normalize the triangular filter weights by the width of the frequecny bands.
	 * \param [in] sampleRate The input scalar in <tt>\ref VX_TYPE_FLOAT32</tt> format containing the sampling rate of the audio data.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppMelFilterBank(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_tensor pDstRoi, vx_scalar freqHigh, vx_scalar freqLow,
															vx_scalar melFormula, vx_scalar nfilter, vx_scalar normalize, vx_scalar sampleRate, vx_scalar inputLayout, vx_scalar outputLayout);

	/*! \brief [Graph] Transpose the input tensor according to the permutation passed
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int<tt> containing the roi values for the input in xywh (w- samples, h - channels) format.
	 * \param [out] pDst The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> format data.
	 * \param [in] pPerm The input array in <tt>\ref VX_TYPE_INT32</tt> format containing the permutation in which transpose needs to be done
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \param [in] roiType The input scalar in <tt>\ref VX_TYPE_INT32</tt> format containing the roi type which can be ltrb/xywh.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppTranspose(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pPerm, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType);

	/*! \brief [Graph] Computes the natural logarithm of 1 + input element-wise and returns the output.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_INT16</tt> format data.
	 * \param [in] pSrcRoi The input tensor of batch size in <tt>unsigned int</tt> containing the roi values for the input.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_FLOAT32</tt> format data.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of input tensor.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtRppLog1p(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_scalar inputLayout);

	/*! \brief [Graph] Invokes the python execution callback function from rocAL, passes the input to the callback and returns the output.
	 * \ingroup group_amd_rpp
	 * \param [in] graph The handle to the graph.
	 * \param [in] pSrc The input tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> or <tt>\ref VX_TYPE_INT16</tt> format data.
	 * \param [out] pDst The output tensor in <tt>\ref VX_TYPE_UINT8</tt> or <tt>\ref VX_TYPE_FLOAT32</tt> or <tt>\ref VX_TYPE_FLOAT16</tt> or <tt>\ref VX_TYPE_INT8</tt> or <tt>\ref VX_TYPE_INT16</tt> format data.
	 * \param [in] bridgeFnPtr The pointer to the bridge function in <tt>\ref VX_TYPE_UINT64</tt> format.
	 * \param [in] functionId The python function ID in <tt>\ref VX_TYPE_UINT64</tt> format.
	 * \param [in] inputLayout The input layout in <tt>\ref VX_TYPE_INT32</tt> denoting the layout of input tensor.
	 * \param [in] outputLayout The output layout in <tt>\ref VX_TYPE_INT32</tt> denotes the layout of output tensor.
	 * \return A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
	 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
	 */
	SHARED_PUBLIC vx_node VX_API_CALL vxExtPythonFunction(vx_graph graph, vx_tensor pSrc, vx_tensor pDst, vx_scalar bridgeFnPtr, vx_scalar functionId, vx_scalar inputLayout, vx_scalar outputLayout);

#ifdef __cplusplus
}
#endif

#endif //_VX_EXT_RPP_H_

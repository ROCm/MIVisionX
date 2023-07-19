/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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
 * \brief The AMD RPP Extension Library.
 *
 * \defgroup group_amd_rpp Extension: AMD RPP Extension API.
 * \brief AMD OpenVX RPP Extension to use as the low-level library for rocAL.
 */

#ifndef dimof
#define dimof(x) (sizeof(x)/sizeof(x[0]))
#endif

#if _WIN32
#define SHARED_PUBLIC __declspec(dllexport)
#else
#define SHARED_PUBLIC __attribute__ ((visibility ("default")))
#endif

vx_node vxCreateNodeByStructure(vx_graph graph, vx_enum kernelenum, vx_reference params[], vx_uint32 num);

#ifdef __cplusplus
extern "C" {
#endif

/*!***********************************************************************************************************
               		         RPP VX_API_ENTRY C Function NODE
*************************************************************************************************************/
/*! \brief [Graph] Creates a RPP Absolute Difference function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc1 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] pSrc2 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AbsoluteDifferencebatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Accumulate function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [inout] pSrc1 The bidirectional image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data that acts as the first input and output.
 * \param [in] pSrc2 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AccumulatebatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Accumulate Squared function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [inout] pSrc The bidirectional image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data that acts as the input and output.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AccumulateSquaredbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Accumulate Weighted function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [inout] pSrc1 The bidirectional image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data that acts as the first input and output.
 * \param [in] pSrc2 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [in] alpha The input array in <tt>\ref VX_TYPE_FLOAT32<tt> format containing the alpha data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AccumulateWeightedbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_array alpha,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Add function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc1 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] pSrc2 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AddbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Bitwise And function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc1 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] pSrc2 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BitwiseANDbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Bitwise NOT function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BitwiseNOTbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Blend function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc1 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] pSrc2 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] alpha The input array in <tt>\ref VX_TYPE_FLOAT32<tt> format containing the alpha data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BlendbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array alpha,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Blur function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] kernelSize The input array in <tt>\ref VX_TYPE_UINT32<tt> format containing the kernel size data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BlurbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Box Filter function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] kernelSize The input array in <tt>\ref VX_TYPE_UINT32<tt> format containing the kernel size data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BoxFilterbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Brightness function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] alpha The input array in <tt>\ref VX_TYPE_FLOAT32<tt> format containing the alpha data.
 * \param [in] beta The input array in <tt>\ref VX_TYPE_FLOAT32<tt> format containing the beta data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BrightnessbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array alpha,vx_array beta,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Canny Edge Detector function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] max The input array in <tt>unsigned char<tt> format containing the max data.
 * \param [in] min The input array in <tt>unsigned char<tt> format containing the min data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_CannyEdgeDetector(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array max,vx_array min,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Channel Combine function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc1 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] pSrc2 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [in] pSrc3 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [out] pDst The output image data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ChannelCombinebatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pSrc3,vx_image pDst,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Channel Extract function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] extractChannelNumber The input array in <tt>\ref VX_TYPE_UINT32<tt> format containing the data for channel number to be extracted.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ChannelExtractbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array extractChannelNumber,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Color Temperature function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] adjustmentValue The input array in <tt>\ref VX_TYPE_UINT32<tt> format containing the data for the adjustment value.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ColorTemperaturebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array adjustmentValue,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Color Twist function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] alpha The input array in <tt>\ref VX_TYPE_FLOAT32<tt> format containing the alpha data.
 * \param [in] beta The input array in <tt>\ref VX_TYPE_FLOAT32<tt> format containing the beta data.
 * \param [in] hue The input array in <tt>\ref VX_TYPE_FLOAT32<tt> format containing the hue data.
 * \param [in] sat The input array in <tt>\ref VX_TYPE_FLOAT32<tt> format containing the saturation data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ColorTwistbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array alpha,vx_array beta,vx_array hue, vx_array sat,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Contrast function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] min The input array in <tt>\ref VX_TYPE_UINT32<tt> format containing the min data.
 * \param [in] max The input array in <tt>\ref VX_TYPE_UINT32<tt> format containing the max data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ContrastbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array min,vx_array max,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Copy function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [out] pDst The output image data.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Copy(vx_graph graph, vx_image pSrc, vx_image pDst);

/*!
* \brief [Graph] Creates a RPP Crop Mirror Normalize function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_CropMirrorNormalizebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1,vx_array mean, vx_array std_dev, vx_array flip, vx_scalar chnShift,vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Crop function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_CropPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1, vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Custom Convolution Normalize function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_CustomConvolutionbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernel,vx_array kernelWidth,vx_array kernelHeight,vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Data Object Copy function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_DataObjectCopybatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Dilate function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] kernelSize The input array in <tt>\ref VX_TYPE_UINT32<tt> format containing the kernel size data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_DilatebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Erade function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] kernelSize The input array in <tt>\ref VX_TYPE_UINT32<tt> format containing the kernel size data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ErodebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ExclusiveORbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Exposure function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] exposureValue The input array in <tt>\ref VX_TYPE_FLOAT32<tt> format containing the exposure value data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ExposurebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array exposureValue,vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Fast Corner Detector function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_FastCornerDetector(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array noOfPixels,vx_array threshold,vx_array nonMaxKernelSize,vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Fish Eye function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_FisheyebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Flip function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] flipAxis The input array in <tt>\ref VX_TYPE_FLOAT32<tt> format containing the flip axis data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_FlipbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array flipAxis,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Fog function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] fogValue The input array in <tt>\ref VX_TYPE_FLOAT32<tt> format containing the fog value data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_FogbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array fogValue,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Gamma Correction function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] gamma The input array in <tt>\ref VX_TYPE_FLOAT32<tt> format containing the gamma data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_GammaCorrectionbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array gamma,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Gaussian Filter function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] stdDev The input array in <tt>\ref VX_TYPE_FLOAT32<tt> format containing the standard deviation data.
 * \param [in] kernelSize The input array in <tt>\ref VX_TYPE_UINT32<tt> format containing the kernel size data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_GaussianFilterbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array stdDev,vx_array kernelSize,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Gaussian Image Pyramid function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] stdDev The input array in <tt>\ref VX_TYPE_FLOAT32<tt> format containing the standard deviation data.
 * \param [in] kernelSize The input array in <tt>\ref VX_TYPE_UINT32<tt> format containing the kernel size data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_GaussianImagePyramidbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array stdDev,vx_array kernelSize,vx_uint32 nbatchSize);
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_HarrisCornerDetector(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array gaussianKernelSize,vx_array stdDev,vx_array kernelSize,vx_array kValue,vx_array threshold,vx_array nonMaxKernelSize,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Gaussian Image Pyramid function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [inout] pSrc The bidirectional image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data that acts as the input and output..
 * \param [in] outputHistogram The input array of given size in <tt>unsigned int<tt> containing the output histogram data.
 * \param [in] bins The input scalar in <tt>unsigned int<tt> to set bins value.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Histogram(vx_graph graph,vx_image pSrc,vx_array outputHistogram,vx_scalar bins);

/*! \brief [Graph] Creates a RPP Histogram Balance function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_HistogramBalancebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Histogram Equalize function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_HistogramEqualizebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Gamma Correction function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] hueShift The input array in <tt>\ref VX_TYPE_FLOAT32<tt> format containing the hue shift data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_HuebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array hueShift,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Inclusive Or function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc1 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] pSrc2 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] alpha The input array in <tt>\ref VX_TYPE_FLOAT32<tt> format containing the alpha data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_InclusiveORbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Jitter function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] kernelSize The input array in <tt>\ref VX_TYPE_UINT32<tt> format containing the kernel size data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_JitterbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Laplacian Image Pyramid function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] stdDev The input array in <tt>float<tt> format containing the standard deviation data.
 * \param [in] kernelSize The input array in <tt>\ref VX_TYPE_UINT32<tt> format containing the kernel size data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_LaplacianImagePyramid(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array stdDev,vx_array kernelSize,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Lens Correction function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] strength The input array in <tt>\ref VX_TYPE_FLOAT32<tt> format containing the strength data.
 * \param [in] zoom The input array in <tt>\ref VX_TYPE_FLOAT32<tt> format containing the zoom data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_LensCorrectionbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array strength,vx_array zoom,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Local Binary Pattern function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_LocalBinaryPatternbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Lookup Table function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] lutPtr The input array in <tt>unsigned char<tt> format containing the strength data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_LookUpTablebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array lutPtr,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Magnitude function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc1 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] pSrc2 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MagnitudebatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Max function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc1 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] pSrc2 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MaxbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Mean Standard Deviation function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MeanStddev(vx_graph graph,vx_image pSrc,vx_scalar mean,vx_scalar stdDev);

/*! \brief [Graph] Creates a RPP Median Filter function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] kernelSize The input array in <tt>\ref VX_TYPE_UINT32<tt> format containing the kernel size data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MedianFilterbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Min function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc1 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] pSrc2 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MinbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Min Max Location function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MinMaxLoc(vx_graph graph,vx_image pSrc,vx_scalar min,vx_scalar max,vx_scalar minLoc,vx_scalar maxLoc);

/*! \brief [Graph] Creates a RPP Multiply function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc1 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] pSrc2 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MultiplybatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_NoisebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array noiseProbability,vx_uint32 nbatchSize);
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_NonLinearFilterbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_NonMaxSupressionbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP NOP function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [out] pDst The output image data.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Nop(vx_graph graph, vx_image pSrc, vx_image pDst);

/*!
* \brief [Graph] Creates a RPP Phase function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_PhasebatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Pixelate function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_PixelatebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Rain function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_RainbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array rainValue,vx_array rainWidth,vx_array rainHeight,vx_array rainTransperancy,vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Random Crop Letter Box function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_RandomCropLetterBoxbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1,vx_array x2,vx_array y2,vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Shadow function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_RandomShadowbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array x1,vx_array y1,vx_array x2,vx_array y2,vx_array numberOfShadows,vx_array maxSizeX,vx_array maxSizeY,vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Remap function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_remap(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array rowRemap,vx_array colRemap,vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Resize function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ResizebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Resize Crop function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ResizeCropbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1,vx_array x2,vx_array y2,vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Resize Crop Mirror function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ResizeCropMirrorPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1,vx_array x2,vx_array y2, vx_array mirrorFlag, vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Resize Mirror Normalize function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ResizeMirrorNormalizeTensor(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array mean,vx_array std_dev,vx_array flip,vx_scalar chnShift,vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Rotate function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_RotatebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array angle,vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Saturation function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_SaturationbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array saturationFactor,vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Scale function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ScalebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array percentage,vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Snow function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_SnowbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array snowValue,vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Sobel function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_SobelbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array sobelType,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Subtract function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc1 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] pSrc2 The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_SubtractbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Tensor Add function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_TensorAdd(vx_graph graph,vx_array pSrc1,vx_array pSrc2,vx_array pDst,vx_scalar tensorDimensions,vx_array tensorDimensionValues);

/*!
* \brief [Graph] Creates a RPP Tensor Lookup function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_TensorLookup(vx_graph graph,vx_array pSrc,vx_array pDst,vx_array lutPtr,vx_scalar tensorDimensions,vx_array tensorDimensionValues);

/*!
* \brief [Graph] Creates a RPP Tensor Matrix Multiply function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_TensorMatrixMultiply(vx_graph graph,vx_array pSrc1,vx_array pSrc2,vx_array pDst,vx_array tensorDimensionValues1,vx_array tensorDimensionValues2);

/*!
* \brief [Graph] Creates a RPP Tensor Multiply function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_TensorMultiply(vx_graph graph,vx_array pSrc1,vx_array pSrc2,vx_array pDst,vx_scalar tensorDimensions,vx_array tensorDimensionValues);

/*!
* \brief [Graph] Creates a RPP Tensor Subtract function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_TensorSubtract(vx_graph graph,vx_array pSrc1,vx_array pSrc2,vx_array pDst,vx_scalar tensorDimensions,vx_array tensorDimensionValues);

/*!
* \brief [Graph] Creates a RPP Threshold function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ThresholdingbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array min,vx_array max,vx_uint32 nbatchSize);

/*! \brief [Graph] Creates a RPP Max function node.
 * \ingroup group_amd_rpp
 * \param [in] graph The handle to the graph.
 * \param [in] pSrc The input image in <tt>\ref VX_DF_IMAGE_U8</tt> or <tt>\ref VX_DF_IMAGE_RGB</tt> format data.
 * \param [in] srcImgWidth The input array of batch size in <tt>unsigned int<tt> containing the image width data.
 * \param [in] srcImgHeight The input array of batch size in <tt>unsigned int<tt> containing the image height data.
 * \param [out] pDst The output image data.
 * \param [in] stdDev The input array in <tt>VX_TYPE_FLOAT32<tt> format containing the standard deviation data.
 * \param [in] nbatchSize The input scalar in <tt>\ref VX_TYPE_UINT32<tt> to set batch size.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_VignettebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array stdDev,vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Warp Affine function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_WarpAffinebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array affine,vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Warp Perspective function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_WarpPerspectivebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array perspective,vx_uint32 nbatchSize);

/*!
* \brief [Graph] Creates a RPP Sequence Rearrange function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_SequenceRearrange(vx_graph graph,vx_image pSrc,vx_image pDst, vx_array newOrder,vx_uint32 newSequenceLength, vx_uint32 sequenceLength, vx_uint32 sequenceCount);

/*!
* \brief [Graph] Creates a RPP Resize Tensor function node.
* \ingroup group_amd_rpp
* @note - TBD
*/
SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Resizetensor(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_int32 interpolation_type, vx_uint32 nbatchSize);
#ifdef __cplusplus
}
#endif

#endif //_VX_EXT_RPP_H_

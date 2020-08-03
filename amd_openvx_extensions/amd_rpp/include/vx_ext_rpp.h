/*
Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.

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
extern  "C" {
#endif

/*!***********************************************************************************************************
               		         RPP VX_API_ENTRY C Function NODE
*************************************************************************************************************/

extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AbsoluteDifference(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AbsoluteDifferencebatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AbsoluteDifferencebatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AbsoluteDifferencebatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Accumulate(vx_graph graph,vx_image pSrc1,vx_image pSrc2);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AccumulatebatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AccumulatebatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AccumulatebatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AccumulateSquared(vx_graph graph,vx_image pSrc);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AccumulateSquaredbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AccumulateSquaredbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AccumulateSquaredbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AccumulateWeighted(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_scalar alpha);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AccumulateWeightedbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_array alpha,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AccumulateWeightedbatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_array alpha,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AccumulateWeightedbatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_scalar alpha,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Add(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AddbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AddbatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AddbatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BilateralFilter(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar kernelSize,vx_scalar sigmaI,vx_scalar sigmaS);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BilateralFilterbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_array sigmaI,vx_array sigmaS,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BilateralFilterbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_array sigmaI,vx_array sigmaS,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BilateralFilterbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar kernelSize,vx_scalar sigmaI,vx_scalar sigmaS,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BitwiseAND(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BitwiseANDbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BitwiseANDbatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BitwiseANDbatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BitwiseNOT(vx_graph graph,vx_image pSrc,vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BitwiseNOTbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BitwiseNOTbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BitwiseNOTbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Blend(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst,vx_scalar alpha);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BlendbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array alpha,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BlendbatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array alpha,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BlendbatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar alpha,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Blur(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar kernelSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BlurbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BlurbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BlurbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BoxFilter(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar kernelSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BoxFilterbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BoxFilterbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BoxFilterbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Brightness(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar alpha,vx_scalar beta);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BrightnessbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array alpha,vx_array beta,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BrightnessbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array alpha,vx_array beta,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BrightnessbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar alpha,vx_scalar beta,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_CannyEdgeDetector(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar max,vx_scalar min);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ChannelCombine(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pSrc3,vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ChannelCombinebatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pSrc3,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ChannelCombinebatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pSrc3,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ChannelExtract(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar extractChannelNumber);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ChannelExtractbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array extractChannelNumber,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ChannelExtractbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar extractChannelNumber,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ColorTemperature(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar adjustmentValue);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ColorTemperaturebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array adjustmentValue,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ColorTemperaturebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array adjustmentValue,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ColorTemperaturebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar adjustmentValue,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ColorTwist(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar alpha,vx_scalar beta,vx_scalar hue,vx_scalar sat);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ColorTwistbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array alpha,vx_array beta,vx_array hue, vx_array sat,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Contrast(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar min,vx_scalar max);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ContrastbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array min,vx_array max,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ContrastbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array min,vx_array max,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ContrastbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar min,vx_scalar max,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ControlFlow(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst,vx_scalar type);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ControlFlowbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array type,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ControlFlowbatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array type,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ControlFlowbatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar type,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Copy(vx_graph graph, vx_image pSrc, vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_CropMirrorNormalizebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1,vx_array mean, vx_array std_dev, vx_array flip, vx_scalar chnShift,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_CropPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1, vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_CustomConvolution(vx_graph graph,vx_image pSrc,vx_image pDst,vx_array kernel,vx_scalar kernelWidth,vx_scalar kernelHeight);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_CustomConvolutionbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernel,vx_array kernelWidth,vx_array kernelHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_CustomConvolutionbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernel,vx_array kernelWidth,vx_array kernelHeight,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_CustomConvolutionbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernel,vx_scalar kernelWidth,vx_scalar kernelHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_DataObjectCopy(vx_graph graph,vx_image pSrc,vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_DataObjectCopybatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_DataObjectCopybatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_DataObjectCopybatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Dilate(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar kernelSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_DilatebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_DilatebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_DilatebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Erode(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar kernelSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ErodebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ErodebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ErodebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ExclusiveOR(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ExclusiveORbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ExclusiveORbatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ExclusiveORbatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Exposure(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar exposureValue);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ExposurebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array exposureValue,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ExposurebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array exposureValue,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ExposurebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar exposureValue,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_FastCornerDetector(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar noOfPixels,vx_scalar threshold,vx_scalar nonMaxKernelSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Fisheye(vx_graph graph,vx_image pSrc,vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_FisheyebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_FisheyebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_FisheyebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Flip(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar flipAxis);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_FlipbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array flipAxis,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_FlipbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array flipAxis,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_FlipbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar flipAxis,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Fog(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar fogValue);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_FogbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array fogValue,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_FogbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array fogValue,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_FogbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar fogValue,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_GammaCorrection(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar gamma);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_GammaCorrectionbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array gamma,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_GammaCorrectionbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array gamma,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_GammaCorrectionbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar gamma,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_GaussianFilter(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar stdDev,vx_scalar kernelSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_GaussianFilterbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array stdDev,vx_array kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_GaussianFilterbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array stdDev,vx_array kernelSize,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_GaussianFilterbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar stdDev,vx_scalar kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_GaussianImagePyramid(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar stdDev,vx_scalar kernelSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_GaussianImagePyramidbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array stdDev,vx_array kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_GaussianImagePyramidbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar stdDev,vx_scalar kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_HarrisCornerDetector(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar gaussianKernelSize,vx_scalar stdDev,vx_scalar kernelSize,vx_scalar kValue,vx_scalar threshold,vx_scalar nonMaxKernelSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Histogram(vx_graph graph,vx_image pSrc,vx_array outputHistogram,vx_scalar bins);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_HistogramBalance(vx_graph graph,vx_image pSrc,vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_HistogramBalancebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_HistogramBalancebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_HistogramBalancebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_HistogramEqualize(vx_graph graph,vx_image pSrc,vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_HistogramEqualizebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_HistogramEqualizebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_HistogramEqualizebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Hue(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar hueShift);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_HuebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array hueShift,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_HuebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array hueShift,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_HuebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar hueShift,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_InclusiveOR(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_InclusiveORbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_InclusiveORbatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_InclusiveORbatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Jitter(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar kernelSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_JitterbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_JitterbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_JitterbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_LaplacianImagePyramid(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar stdDev,vx_scalar kernelSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_LensCorrection(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar strength,vx_scalar zoom);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_LensCorrectionbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array strength,vx_array zoom,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_LensCorrectionbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array strength,vx_array zoom,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_LensCorrectionbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar strength,vx_scalar zoom,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_LocalBinaryPattern(vx_graph graph,vx_image pSrc,vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_LocalBinaryPatternbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_LocalBinaryPatternbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_LocalBinaryPatternbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_LookUpTable(vx_graph graph,vx_image pSrc,vx_image pDst,vx_array lutPtr);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_LookUpTablebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array lutPtr,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_LookUpTablebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array lutPtr,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_LookUpTablebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array lutPtr,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Magnitude(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MagnitudebatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MagnitudebatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MagnitudebatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Max(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MaxbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MaxbatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MaxbatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MeanStddev(vx_graph graph,vx_image pSrc,vx_scalar mean,vx_scalar stdDev);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MedianFilter(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar kernelSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MedianFilterbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MedianFilterbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MedianFilterbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Min(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MinbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MinbatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MinbatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MinMaxLoc(vx_graph graph,vx_image pSrc,vx_scalar min,vx_scalar max,vx_scalar minLoc,vx_scalar maxLoc);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Multiply(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MultiplybatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MultiplybatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MultiplybatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Noise(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar noiseProbability);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_NoisebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array noiseProbability,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_NoisebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array noiseProbability,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_NoisebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar noiseProbability,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_NonLinearFilter(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar kernelSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_NonLinearFilterbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_NonLinearFilterbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_NonLinearFilterbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_NonMaxSupression(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar kernelSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_NonMaxSupressionbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_NonMaxSupressionbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_NonMaxSupressionbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Nop(vx_graph graph, vx_image pSrc, vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Occlusion(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst,vx_scalar src1x1,vx_scalar src1y1,vx_scalar src1x2,vx_scalar src1y2,vx_scalar src2x1,vx_scalar src2y1,vx_scalar src2x2,vx_scalar src2y2);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_OcclusionbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array src1x1,vx_array src1y1,vx_array src1x2,vx_array src1y2,vx_array src2x1,vx_array src2y1,vx_array src2x2,vx_array src2y2,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_OcclusionbatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array src1x1,vx_array src1y1,vx_array src1x2,vx_array src1y2,vx_array src2x1,vx_array src2y1,vx_array src2x2,vx_array src2y2,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_OcclusionbatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar src1x1,vx_scalar src1y1,vx_scalar src1x2,vx_scalar src1y2,vx_scalar src2x1,vx_scalar src2y1,vx_scalar src2x2,vx_scalar src2y2,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Phase(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_PhasebatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_PhasebatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_PhasebatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Pixelate(vx_graph graph,vx_image pSrc,vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_PixelatebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_PixelatebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_PixelatebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Rain(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar rainValue,vx_scalar rainWidth,vx_scalar rainHeight,vx_scalar rainTransperancy);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_RainbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array rainValue,vx_array rainWidth,vx_array rainHeight,vx_array rainTransperancy,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_RainbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array rainValue,vx_array rainWidth,vx_array rainHeight,vx_array rainTransperancy,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_RainbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar rainValue,vx_scalar rainWidth,vx_scalar rainHeight,vx_scalar rainTransperancy,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_RandomCropLetterBox(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar x1,vx_scalar y1,vx_scalar x2,vx_scalar y2);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_RandomCropLetterBoxbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1,vx_array x2,vx_array y2,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_RandomCropLetterBoxbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1,vx_array x2,vx_array y2,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_RandomCropLetterBoxbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar dstImgWidth,vx_scalar dstImgHeight,vx_scalar x1,vx_scalar y1,vx_scalar x2,vx_scalar y2,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_RandomShadow(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar x1,vx_scalar y1,vx_scalar x2,vx_scalar y2,vx_scalar numberOfShadows,vx_scalar maxSizeX,vx_scalar maxSizeY);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_RandomShadowbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array x1,vx_array y1,vx_array x2,vx_array y2,vx_array numberOfShadows,vx_array maxSizeX,vx_array maxSizeY,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_RandomShadowbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array x1,vx_array y1,vx_array x2,vx_array y2,vx_array numberOfShadows,vx_array maxSizeX,vx_array maxSizeY,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_RandomShadowbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar x1,vx_scalar y1,vx_scalar x2,vx_scalar y2,vx_scalar numberOfShadows,vx_scalar maxSizeX,vx_scalar maxSizeY,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_remap(vx_graph graph,vx_image pSrc,vx_image pDst,vx_array rowRemap,vx_array colRemap);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Resize(vx_graph graph,vx_image pSrc,vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ResizebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ResizebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ResizebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar dstImgWidth,vx_scalar dstImgHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ResizeCrop(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar x1,vx_scalar y1,vx_scalar x2,vx_scalar y2);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ResizeCropbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1,vx_array x2,vx_array y2,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ResizeCropbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1,vx_array x2,vx_array y2,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ResizeCropbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar dstImgWidth,vx_scalar dstImgHeight,vx_scalar x1,vx_scalar y1,vx_scalar x2,vx_scalar y2,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ResizeCropMirrorPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1,vx_array x2,vx_array y2, vx_array mirrorFlag, vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Rotate(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar angle);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_RotatebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array angle,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_RotatebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array angle,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_RotatebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar dstImgWidth,vx_scalar dstImgHeight,vx_scalar angle,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Saturation(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar saturationFactor);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_SaturationbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array saturationFactor,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_SaturationbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array saturationFactor,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_SaturationbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar saturationFactor,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Scale(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar percentage);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ScalebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array percentage,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ScalebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array percentage,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ScalebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar dstImgWidth,vx_scalar dstImgHeight,vx_scalar percentage,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Snow(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar snowValue);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_SnowbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array snowValue,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_SnowbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array snowValue,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_SnowbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar snowValue,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Sobel(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar sobelType);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_SobelbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array sobelType,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_SobelbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array sobelType,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_SobelbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar sobelType,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Subtract(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_SubtractbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_SubtractbatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_SubtractbatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_TensorAdd(vx_graph graph,vx_array pSrc1,vx_array pSrc2,vx_array pDst,vx_scalar tensorDimensions,vx_array tensorDimensionValues);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_TensorLookup(vx_graph graph,vx_array pSrc,vx_array pDst,vx_array lutPtr,vx_scalar tensorDimensions,vx_array tensorDimensionValues);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_TensorMatrixMultiply(vx_graph graph,vx_array pSrc1,vx_array pSrc2,vx_array pDst,vx_array tensorDimensionValues1,vx_array tensorDimensionValues2);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_TensorMultiply(vx_graph graph,vx_array pSrc1,vx_array pSrc2,vx_array pDst,vx_scalar tensorDimensions,vx_array tensorDimensionValues);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_TensorSubtract(vx_graph graph,vx_array pSrc1,vx_array pSrc2,vx_array pDst,vx_scalar tensorDimensions,vx_array tensorDimensionValues);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Thresholding(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar min,vx_scalar max);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ThresholdingbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array min,vx_array max,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ThresholdingbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array min,vx_array max,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ThresholdingbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar min,vx_scalar max,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Vignette(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar stdDev);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_VignettebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array stdDev,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_VignettebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array stdDev,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_VignettebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar stdDev,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_WarpAffine(vx_graph graph,vx_image pSrc,vx_image pDst,vx_array affine);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_WarpAffinebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array affine,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_WarpAffinebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array affine,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_WarpAffinebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar dstImgWidth,vx_scalar dstImgHeight,vx_array affine,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_WarpPerspective(vx_graph graph,vx_image pSrc,vx_image pDst,vx_array perspective);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_WarpPerspectivebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array perspective,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_WarpPerspectivebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array perspective,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_WarpPerspectivebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar dstImgWidth,vx_scalar dstImgHeight,vx_array perspective,vx_uint32 nbatchSize);

#ifdef __cplusplus
}
#endif

#endif //_VX_EXT_RPP_H_

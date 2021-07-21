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

extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AbsoluteDifferencebatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AccumulatebatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AccumulateSquaredbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AccumulateWeightedbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_array alpha,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_AddbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BilateralFilterbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_array sigmaI,vx_array sigmaS,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BitwiseANDbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BitwiseNOTbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BlendbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array alpha,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BlurbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BoxFilterbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_BrightnessbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array alpha,vx_array beta,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_CannyEdgeDetector(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar max,vx_scalar min);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ChannelCombinebatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pSrc3,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ChannelExtractbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array extractChannelNumber,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ColorTemperaturebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array adjustmentValue,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ColorTwistbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array alpha,vx_array beta,vx_array hue, vx_array sat,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ContrastbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array min,vx_array max,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Copy(vx_graph graph, vx_image pSrc, vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_CropMirrorNormalizebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1,vx_array mean, vx_array std_dev, vx_array flip, vx_scalar chnShift,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_CropPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1, vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_CustomConvolutionbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernel,vx_array kernelWidth,vx_array kernelHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_DataObjectCopybatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_DilatebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ErodebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ExclusiveORbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ExposurebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array exposureValue,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_FastCornerDetector(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar noOfPixels,vx_scalar threshold,vx_scalar nonMaxKernelSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_FisheyebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_FlipbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array flipAxis,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_FogbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array fogValue,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_GammaCorrectionbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array gamma,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_GaussianFilterbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array stdDev,vx_array kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_GaussianImagePyramidbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array stdDev,vx_array kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_HarrisCornerDetector(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar gaussianKernelSize,vx_scalar stdDev,vx_scalar kernelSize,vx_scalar kValue,vx_scalar threshold,vx_scalar nonMaxKernelSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Histogram(vx_graph graph,vx_image pSrc,vx_array outputHistogram,vx_scalar bins);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_HistogramBalancebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_HistogramEqualizebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_HuebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array hueShift,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_InclusiveORbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_JitterbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_LaplacianImagePyramid(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar stdDev,vx_scalar kernelSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_LensCorrectionbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array strength,vx_array zoom,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_LocalBinaryPatternbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_LookUpTablebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array lutPtr,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MagnitudebatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MaxbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MeanStddev(vx_graph graph,vx_image pSrc,vx_scalar mean,vx_scalar stdDev);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MedianFilterbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MinbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MinMaxLoc(vx_graph graph,vx_image pSrc,vx_scalar min,vx_scalar max,vx_scalar minLoc,vx_scalar maxLoc);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_MultiplybatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_NoisebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array noiseProbability,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_NonLinearFilterbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_NonMaxSupressionbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_Nop(vx_graph graph, vx_image pSrc, vx_image pDst);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_PhasebatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_PixelatebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_RainbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array rainValue,vx_array rainWidth,vx_array rainHeight,vx_array rainTransperancy,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_RandomCropLetterBoxbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1,vx_array x2,vx_array y2,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_RandomShadowbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array x1,vx_array y1,vx_array x2,vx_array y2,vx_array numberOfShadows,vx_array maxSizeX,vx_array maxSizeY,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_remap(vx_graph graph,vx_image pSrc,vx_image pDst,vx_array rowRemap,vx_array colRemap);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ResizebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ResizeCropbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1,vx_array x2,vx_array y2,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ResizeCropMirrorPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1,vx_array x2,vx_array y2, vx_array mirrorFlag, vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_RotatebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array angle,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_SaturationbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array saturationFactor,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ScalebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array percentage,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_SnowbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array snowValue,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_SobelbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array sobelType,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_SubtractbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_TensorAdd(vx_graph graph,vx_array pSrc1,vx_array pSrc2,vx_array pDst,vx_scalar tensorDimensions,vx_array tensorDimensionValues);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_TensorLookup(vx_graph graph,vx_array pSrc,vx_array pDst,vx_array lutPtr,vx_scalar tensorDimensions,vx_array tensorDimensionValues);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_TensorMatrixMultiply(vx_graph graph,vx_array pSrc1,vx_array pSrc2,vx_array pDst,vx_array tensorDimensionValues1,vx_array tensorDimensionValues2);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_TensorMultiply(vx_graph graph,vx_array pSrc1,vx_array pSrc2,vx_array pDst,vx_scalar tensorDimensions,vx_array tensorDimensionValues);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_TensorSubtract(vx_graph graph,vx_array pSrc1,vx_array pSrc2,vx_array pDst,vx_scalar tensorDimensions,vx_array tensorDimensionValues);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_ThresholdingbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array min,vx_array max,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_VignettebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array stdDev,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_WarpAffinebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array affine,vx_uint32 nbatchSize);
extern  "C" SHARED_PUBLIC vx_node VX_API_CALL vxExtrppNode_WarpPerspectivebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array perspective,vx_uint32 nbatchSize);

#ifdef __cplusplus
}
#endif

#endif //_VX_EXT_RPP_H_

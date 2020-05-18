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


#ifndef __ago_kernels_api_h__
#define __ago_kernels_api_h__

#include "ago_internal.h"

// import all kernels into framework
int agoPublishKernels(AgoContext * acontext);

// OpenVX 1.x built-in kernels
int ovxKernel_Invalid(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_ColorConvert(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_ChannelExtract(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_ChannelCombine(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_Sobel3x3(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_Magnitude(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_Phase(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_ScaleImage(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_TableLookup(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_Histogram(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_EqualizeHistogram(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_AbsDiff(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_MeanStdDev(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_Threshold(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_IntegralImage(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_Dilate3x3(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_Erode3x3(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_Median3x3(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_Box3x3(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_Gaussian3x3(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_CustomConvolution(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_GaussianPyramid(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_Accumulate(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_AccumulateWeighted(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_AccumulateSquare(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_MinMaxLoc(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_ConvertDepth(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_CannyEdgeDetector(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_And(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_Or(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_Xor(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_Not(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_Multiply(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_Add(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_Subtract(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_WarpAffine(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_WarpPerspective(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_HarrisCorners(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_FastCorners(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_OpticalFlowPyrLK(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_Remap(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_HalfScaleGaussian(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_Copy(AgoNode * node, AgoKernelCommand cmd);
int ovxKernel_Select(AgoNode * node, AgoKernelCommand cmd);

// AMD low-level kernels
int agoKernel_Set00_U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_SetFF_U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Not_U8_U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Not_U8_U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Not_U1_U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Not_U1_U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Lut_U8_U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Threshold_U8_U8_Binary(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Threshold_U8_U8_Range(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Threshold_U1_U8_Binary(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Threshold_U1_U8_Range(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ThresholdNot_U8_U8_Binary(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ThresholdNot_U8_U8_Range(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ThresholdNot_U1_U8_Binary(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ThresholdNot_U1_U8_Range(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorDepth_U8_S16_Wrap(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorDepth_U8_S16_Sat(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorDepth_S16_U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Add_U8_U8U8_Wrap(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Add_U8_U8U8_Sat(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Sub_U8_U8U8_Wrap(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Sub_U8_U8U8_Sat(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Mul_U8_U8U8_Wrap_Trunc(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Mul_U8_U8U8_Wrap_Round(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Mul_U8_U8U8_Sat_Trunc(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Mul_U8_U8U8_Sat_Round(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_And_U8_U8U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_And_U8_U8U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_And_U8_U1U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_And_U8_U1U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_And_U1_U8U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_And_U1_U8U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_And_U1_U1U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_And_U1_U1U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Or_U8_U8U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Or_U8_U8U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Or_U8_U1U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Or_U8_U1U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Or_U1_U8U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Or_U1_U8U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Or_U1_U1U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Or_U1_U1U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Xor_U8_U8U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Xor_U8_U8U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Xor_U8_U1U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Xor_U8_U1U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Xor_U1_U8U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Xor_U1_U8U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Xor_U1_U1U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Xor_U1_U1U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Nand_U8_U8U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Nand_U8_U8U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Nand_U8_U1U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Nand_U8_U1U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Nand_U1_U8U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Nand_U1_U8U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Nand_U1_U1U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Nand_U1_U1U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Nor_U8_U8U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Nor_U8_U8U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Nor_U8_U1U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Nor_U8_U1U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Nor_U1_U8U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Nor_U1_U8U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Nor_U1_U1U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Nor_U1_U1U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Xnor_U8_U8U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Xnor_U8_U8U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Xnor_U8_U1U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Xnor_U8_U1U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Xnor_U1_U8U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Xnor_U1_U8U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Xnor_U1_U1U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Xnor_U1_U1U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_AbsDiff_U8_U8U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_AccumulateWeighted_U8_U8U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Add_S16_U8U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Sub_S16_U8U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Mul_S16_U8U8_Wrap_Trunc(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Mul_S16_U8U8_Wrap_Round(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Mul_S16_U8U8_Sat_Trunc(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Mul_S16_U8U8_Sat_Round(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Add_S16_S16U8_Wrap(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Add_S16_S16U8_Sat(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Accumulate_S16_S16U8_Sat(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Sub_S16_S16U8_Wrap(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Sub_S16_S16U8_Sat(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Mul_S16_S16U8_Wrap_Trunc(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Mul_S16_S16U8_Wrap_Round(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Mul_S16_S16U8_Sat_Trunc(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Mul_S16_S16U8_Sat_Round(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_AccumulateSquared_S16_S16U8_Sat(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Sub_S16_U8S16_Wrap(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Sub_S16_U8S16_Sat(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_AbsDiff_S16_S16S16_Sat(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Add_S16_S16S16_Wrap(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Add_S16_S16S16_Sat(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Sub_S16_S16S16_Wrap(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Sub_S16_S16S16_Sat(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Mul_S16_S16S16_Wrap_Trunc(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Mul_S16_S16S16_Wrap_Round(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Mul_S16_S16S16_Sat_Trunc(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Mul_S16_S16S16_Sat_Round(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Magnitude_S16_S16S16(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Phase_U8_S16S16(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ChannelCopy_U8_U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ChannelCopy_U8_U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ChannelCopy_U1_U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ChannelCopy_U1_U1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ChannelExtract_U8_U16_Pos0(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ChannelExtract_U8_U16_Pos1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ChannelExtract_U8_U24_Pos0(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ChannelExtract_U8_U24_Pos1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ChannelExtract_U8_U24_Pos2(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ChannelExtract_U8_U32_Pos0(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ChannelExtract_U8_U32_Pos1(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ChannelExtract_U8_U32_Pos2(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ChannelExtract_U8_U32_Pos3(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ChannelExtract_U8U8U8_U24(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ChannelExtract_U8U8U8_U32(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ChannelExtract_U8U8U8U8_U32(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ChannelCombine_U16_U8U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ChannelCombine_U24_U8U8U8_RGB(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ChannelCombine_U32_U8U8U8_UYVY(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ChannelCombine_U32_U8U8U8_YUYV(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ChannelCombine_U32_U8U8U8U8_RGBX(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Mul_U24_U24U8_Sat_Round(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Mul_U32_U32U8_Sat_Round(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_RGB_RGBX(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_RGB_UYVY(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_RGB_YUYV(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_RGB_IYUV(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_RGB_NV12(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_RGB_NV21(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_RGBX_RGB(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_RGBX_UYVY(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_RGBX_YUYV(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_RGBX_IYUV(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_RGBX_NV12(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_RGBX_NV21(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_YUV4_RGB(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_YUV4_RGBX(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ScaleUp2x2_U8_U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_FormatConvert_UV_UV12(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_IYUV_RGB(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_IYUV_RGBX(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_FormatConvert_IYUV_UYVY(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_FormatConvert_IYUV_YUYV(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_FormatConvert_IUV_UV12(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_NV12_RGB(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_NV12_RGBX(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_FormatConvert_NV12_UYVY(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_FormatConvert_NV12_YUYV(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_FormatConvert_UV12_IUV(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_Y_RGB(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_Y_RGBX(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_U_RGB(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_U_RGBX(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_V_RGB(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_V_RGBX(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_IU_RGB(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_IU_RGBX(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_IV_RGB(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_IV_RGBX(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_IUV_RGB(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_IUV_RGBX(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_UV12_RGB(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ColorConvert_UV12_RGBX(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Box_U8_U8_3x3(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Dilate_U8_U8_3x3(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Erode_U8_U8_3x3(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Median_U8_U8_3x3(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Gaussian_U8_U8_3x3(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ScaleGaussianHalf_U8_U8_3x3(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ScaleGaussianHalf_U8_U8_5x5(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ScaleGaussianOrb_U8_U8_5x5(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Convolve_U8_U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Convolve_S16_U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_LinearFilter_ANY_ANY(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_LinearFilter_ANYx2_ANY(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_SobelMagnitude_S16_U8_3x3(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_SobelPhase_U8_U8_3x3(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_SobelMagnitudePhase_S16U8_U8_3x3(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Sobel_S16S16_U8_3x3_GXY(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Sobel_S16_U8_3x3_GX(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Sobel_S16_U8_3x3_GY(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Dilate_U1_U8_3x3(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Erode_U1_U8_3x3(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Dilate_U1_U1_3x3(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Erode_U1_U1_3x3(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Dilate_U8_U1_3x3(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Erode_U8_U1_3x3(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_FastCorners_XY_U8_Supression(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_FastCorners_XY_U8_NoSupression(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_HarrisSobel_HG3_U8_3x3(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_HarrisSobel_HG3_U8_5x5(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_HarrisSobel_HG3_U8_7x7(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_HarrisScore_HVC_HG3_3x3(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_HarrisScore_HVC_HG3_5x5(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_HarrisScore_HVC_HG3_7x7(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_CannySobelSuppThreshold_U8_U8_3x3_L1NORM(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_CannySobelSuppThreshold_U8_U8_3x3_L2NORM(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_CannySobelSuppThreshold_U8_U8_5x5_L1NORM(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_CannySobelSuppThreshold_U8_U8_5x5_L2NORM(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_CannySobelSuppThreshold_U8_U8_7x7_L1NORM(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_CannySobelSuppThreshold_U8_U8_7x7_L2NORM(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_CannySobelSuppThreshold_U8XY_U8_3x3_L1NORM(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_CannySobelSuppThreshold_U8XY_U8_3x3_L2NORM(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_CannySobelSuppThreshold_U8XY_U8_5x5_L1NORM(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_CannySobelSuppThreshold_U8XY_U8_5x5_L2NORM(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_CannySobelSuppThreshold_U8XY_U8_7x7_L1NORM(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_CannySobelSuppThreshold_U8XY_U8_7x7_L2NORM(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_CannySobel_U16_U8_3x3_L1NORM(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_CannySobel_U16_U8_3x3_L2NORM(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_CannySobel_U16_U8_5x5_L1NORM(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_CannySobel_U16_U8_5x5_L2NORM(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_CannySobel_U16_U8_7x7_L1NORM(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_CannySobel_U16_U8_7x7_L2NORM(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_CannySuppThreshold_U8_U16_3x3(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_CannySuppThreshold_U8XY_U16_3x3(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_NonMaxSupp_XY_ANY_3x3(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Remap_U8_U8_Nearest(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Remap_U8_U8_Nearest_Constant(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Remap_U8_U8_Bilinear(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Remap_U8_U8_Bilinear_Constant(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Remap_U24_U24_Bilinear(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Remap_U24_U32_Bilinear(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Remap_U32_U32_Bilinear(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_WarpAffine_U8_U8_Nearest(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_WarpAffine_U8_U8_Nearest_Constant(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_WarpAffine_U8_U8_Bilinear(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_WarpAffine_U8_U8_Bilinear_Constant(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_WarpPerspective_U8_U8_Nearest(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_WarpPerspective_U8_U8_Nearest_Constant(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_WarpPerspective_U8_U8_Bilinear(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_WarpPerspective_U8_U8_Bilinear_Constant(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ScaleImage_U8_U8_Nearest(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ScaleImage_U8_U8_Bilinear(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ScaleImage_U8_U8_Bilinear_Replicate(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ScaleImage_U8_U8_Bilinear_Constant(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_ScaleImage_U8_U8_Area(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_OpticalFlowPyrLK_XY_XY(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_OpticalFlowPrepareLK_XY_XY(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_OpticalFlowImageLK_XY_XY(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_OpticalFlowFinalLK_XY_XY(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_HarrisMergeSortAndPick_XY_HVC(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_HarrisMergeSortAndPick_XY_XYS(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_FastCornerMerge_XY_XY(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_CannyEdgeTrace_U8_U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_CannyEdgeTrace_U8_U8XY(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_IntegralImage_U32_U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Histogram_DATA_U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_MeanStdDev_DATA_U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_MinMax_DATA_U8(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_MinMax_DATA_S16(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Equalize_DATA_DATA(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_HistogramMerge_DATA_DATA(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_MeanStdDevMerge_DATA_DATA(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_MinMaxMerge_DATA_DATA(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_MinMaxLoc_DATA_U8DATA_Loc_None_Count_Min(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_MinMaxLoc_DATA_U8DATA_Loc_None_Count_Max(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_MinMaxLoc_DATA_U8DATA_Loc_None_Count_MinMax(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_MinMaxLoc_DATA_U8DATA_Loc_Min_Count_Min(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_MinMaxLoc_DATA_U8DATA_Loc_Min_Count_MinMax(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_MinMaxLoc_DATA_U8DATA_Loc_Max_Count_Max(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_MinMaxLoc_DATA_U8DATA_Loc_Max_Count_MinMax(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_MinMaxLoc_DATA_U8DATA_Loc_MinMax_Count_MinMax(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_MinMaxLoc_DATA_S16DATA_Loc_None_Count_Min(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_MinMaxLoc_DATA_S16DATA_Loc_None_Count_Max(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_MinMaxLoc_DATA_S16DATA_Loc_None_Count_MinMax(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_MinMaxLoc_DATA_S16DATA_Loc_Min_Count_Min(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_MinMaxLoc_DATA_S16DATA_Loc_Min_Count_MinMax(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_MinMaxLoc_DATA_S16DATA_Loc_Max_Count_Max(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_MinMaxLoc_DATA_S16DATA_Loc_Max_Count_MinMax(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_MinMaxLoc_DATA_S16DATA_Loc_MinMax_Count_MinMax(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_MinMaxLocMerge_DATA_DATA(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Copy_DATA_DATA(AgoNode * node, AgoKernelCommand cmd);
int agoKernel_Select_DATA_DATA_DATA(AgoNode * node, AgoKernelCommand cmd);

#endif // __ago_kernels_api_h__

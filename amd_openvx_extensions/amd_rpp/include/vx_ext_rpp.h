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


#ifndef _VX_EXT_RPP_H_
#define _VX_EXT_RPP_H_

#include <VX/vx.h>
#include "kernels_rpp.h"

#if ENABLE_OPENCL
#include <CL/cl.h>
#endif


	/*!***********************************************************************************************************
						RPP VX_API_ENTRY C Function NODE
	*************************************************************************************************************/
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Copy(vx_graph graph, vx_image pSrc, vx_image pDst);
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_brightness(vx_graph graph, vx_image pSrc, vx_image pDst, vx_float32 alpha, vx_int32 beta);
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_contrast(vx_graph graph, vx_image pSrc, vx_image pDst, vx_uint32 max, vx_uint32 min);
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_blur(vx_graph graph, vx_image pSrc, vx_image pDst, vx_float32 sdev);
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Flip(vx_graph graph, vx_image pSrc, vx_image pDst, vx_int32 flipAxis);
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_GammaCorrection(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_float32 alpha);
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Resize(vx_graph graph, vx_image pSrc, vx_image pDst,
                                                                           vx_int32 DestWidth, vx_int32 DestHeight);
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ResizeCrop(vx_graph graph, vx_image pSrc, vx_image pDst,
                                                                    vx_int32 DestWidth, vx_int32 DestHeight, vx_int32 x1,
                                                                    vx_int32 y1, vx_int32 x2, vx_int32 y2);
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Rotate(vx_graph graph, vx_image pSrc, vx_image pDst, vx_int32 DestWidth, vx_int32 DestHeight, vx_float32 angle);

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_WarpAffine(vx_graph graph, vx_image pSrc, vx_image pDst,vx_int32 DestWidth, vx_int32 DestHeight,
                                                                     vx_float32 affineVal1, vx_float32 affineVal2,vx_float32 affineVal3,vx_float32 affineVal4,
                                                                     vx_float32 affineVal5,vx_float32 affineVal6);
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Blend(vx_graph graph, vx_image pSrc1, vx_image pSrc2 , vx_image pDst, vx_float32 alpha);
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Exposure(vx_graph graph, vx_image pSrc1, vx_image pDst, vx_float32 exposureValue);
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Fisheye(vx_graph graph, vx_image pSrc, vx_image pDst);
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Snow(vx_graph graph, vx_image pSrc, vx_image pDst, vx_float32 snowValue);
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Vignette(vx_graph graph, vx_image pSrc, vx_image pDst, vx_float32 stdDev);
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_LensCorrection(vx_graph graph, vx_image pSrc, vx_image pDst, vx_float32 strength, vx_float32 zoom);
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Pixelate(vx_graph graph, vx_image pSrc, vx_image pDst);
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Jitter(vx_graph graph, vx_image pSrc, vx_image pDst, vx_uint32 kernelSize);
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ColorTemperature(vx_graph graph, vx_image pSrc, vx_image pDst, vx_int32 adjustmentValue);
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Rain(vx_graph graph, vx_image pSrc, vx_image pDst, vx_float32 rainValue,vx_uint32 rainWidth, vx_uint32 rainHeight, vx_float32 rainTransparency);
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Fog(vx_graph graph, vx_image pSrc, vx_image pDst, vx_float32 fogValue);
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_NoiseSnp(vx_graph graph, vx_image pSrc, vx_image pDst, vx_float32 noiseProbability);

#endif //_VX_EXT_RPP_H_

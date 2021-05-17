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

#include "internal_publishKernels.h"
#include "vx_ext_rpp.h"

vx_uint32 getGraphAffinity(vx_graph graph)
{
    AgoTargetAffinityInfo affinity;
    vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_AMD_AFFINITY,&affinity, sizeof(affinity));;
    if(affinity.device_type != AGO_TARGET_AFFINITY_GPU && affinity.device_type != AGO_TARGET_AFFINITY_CPU)
        affinity.device_type = AGO_TARGET_AFFINITY_CPU;
   // std::cerr<<"\n affinity "<<affinity.device_type;
    return affinity.device_type;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Brightness(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar alpha,vx_scalar beta)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) alpha,
			(vx_reference) beta,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BRIGHTNESS, params, 5);
	}
	return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BrightnessbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar alpha,vx_scalar beta,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) alpha,
			(vx_reference) beta,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BRIGHTNESSBATCHPS, params, 8);
	}
	return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BrightnessbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array alpha,vx_array beta,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) alpha,
			(vx_reference) beta,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BRIGHTNESSBATCHPD, params, 8);
	}
	return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BrightnessbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array alpha,vx_array beta,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) alpha,
			(vx_reference) beta,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BRIGHTNESSBATCHPDROID, params, 12);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_GammaCorrection(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar gamma)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) gamma,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_GAMMACORRECTION, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_GammaCorrectionbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar gamma,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) gamma,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_GAMMACORRECTIONBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_GammaCorrectionbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array gamma,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) gamma,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_GAMMACORRECTIONBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_GammaCorrectionbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array gamma,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) gamma,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_GAMMACORRECTIONBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Blend(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst,vx_scalar alpha)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) pDst,
			(vx_reference) alpha,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BLEND, params, 5);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BlendbatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar alpha,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) alpha,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BLENDBATCHPS, params, 8);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BlendbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array alpha,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) alpha,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BLENDBATCHPD, params, 8);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BlendbatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array alpha,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) alpha,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BLENDBATCHPDROID, params, 12);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Blur(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar kernelSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BLUR, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BlurbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar kernelSize,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BLURBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BlurbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BLURBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BlurbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BLURBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Contrast(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar min,vx_scalar max)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) min,
			(vx_reference) max,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_CONTRAST, params, 5);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ContrastbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar min,vx_scalar max,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) min,
			(vx_reference) max,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_CONTRASTBATCHPS, params, 8);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ContrastbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array min,vx_array max,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) min,
			(vx_reference) max,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_CONTRASTBATCHPD, params, 8);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ContrastbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array min,vx_array max,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) min,
			(vx_reference) max,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_CONTRASTBATCHPDROID, params, 12);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Pixelate(vx_graph graph,vx_image pSrc,vx_image pDst)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_PIXELATE, params, 3);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_PixelatebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_PIXELATEBATCHPS, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_PixelatebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_PIXELATEBATCHPD, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_PixelatebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_PIXELATEBATCHPDROID, params, 10);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Jitter(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar kernelSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_JITTER, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_JitterbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar kernelSize,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_JITTERBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_JitterbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_JITTERBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_JitterbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_JITTERBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Occlusion(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst,vx_scalar src1x1,vx_scalar src1y1,vx_scalar src1x2,vx_scalar src1y2,vx_scalar src2x1,vx_scalar src2y1,vx_scalar src2x2,vx_scalar src2y2)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) pDst,
			(vx_reference) src1x1,
			(vx_reference) src1y1,
			(vx_reference) src1x2,
			(vx_reference) src1y2,
			(vx_reference) src2x1,
			(vx_reference) src2y1,
			(vx_reference) src2x2,
			(vx_reference) src2y2,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_OCCLUSION, params, 12);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_OcclusionbatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar src1x1,vx_scalar src1y1,vx_scalar src1x2,vx_scalar src1y2,vx_scalar src2x1,vx_scalar src2y1,vx_scalar src2x2,vx_scalar src2y2,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) src1x1,
			(vx_reference) src1y1,
			(vx_reference) src1x2,
			(vx_reference) src1y2,
			(vx_reference) src2x1,
			(vx_reference) src2y1,
			(vx_reference) src2x2,
			(vx_reference) src2y2,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_OCCLUSIONBATCHPS, params, 15);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_OcclusionbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array src1x1,vx_array src1y1,vx_array src1x2,vx_array src1y2,vx_array src2x1,vx_array src2y1,vx_array src2x2,vx_array src2y2,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) src1x1,
			(vx_reference) src1y1,
			(vx_reference) src1x2,
			(vx_reference) src1y2,
			(vx_reference) src2x1,
			(vx_reference) src2y1,
			(vx_reference) src2x2,
			(vx_reference) src2y2,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_OCCLUSIONBATCHPD, params, 17);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_OcclusionbatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array src1x1,vx_array src1y1,vx_array src1x2,vx_array src1y2,vx_array src2x1,vx_array src2y1,vx_array src2x2,vx_array src2y2,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) src1x1,
			(vx_reference) src1y1,
			(vx_reference) src1x2,
			(vx_reference) src1y2,
			(vx_reference) src2x1,
			(vx_reference) src2y1,
			(vx_reference) src2x2,
			(vx_reference) src2y2,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_OCCLUSIONBATCHPDROID, params, 22);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Snow(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar snowValue)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) snowValue,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_SNOW, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_SnowbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar snowValue,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) snowValue,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_SNOWBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_SnowbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array snowValue,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) snowValue,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_SNOWBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_SnowbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array snowValue,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) snowValue,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_SNOWBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Noise(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar noiseProbability)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) noiseProbability,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_NOISE, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_NoisebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar noiseProbability,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) noiseProbability,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_NOISEBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_NoisebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array noiseProbability,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) noiseProbability,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_NOISEBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_NoisebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array noiseProbability,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) noiseProbability,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_NOISEBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_RandomShadow(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar x1,vx_scalar y1,vx_scalar x2,vx_scalar y2,vx_scalar numberOfShadows,vx_scalar maxSizeX,vx_scalar maxSizeY)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) x1,
			(vx_reference) y1,
			(vx_reference) x2,
			(vx_reference) y2,
			(vx_reference) numberOfShadows,
			(vx_reference) maxSizeX,
			(vx_reference) maxSizeY,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_RANDOMSHADOW, params, 10);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_RandomShadowbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar x1,vx_scalar y1,vx_scalar x2,vx_scalar y2,vx_scalar numberOfShadows,vx_scalar maxSizeX,vx_scalar maxSizeY,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) x1,
			(vx_reference) y1,
			(vx_reference) x2,
			(vx_reference) y2,
			(vx_reference) numberOfShadows,
			(vx_reference) maxSizeX,
			(vx_reference) maxSizeY,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_RANDOMSHADOWBATCHPS, params, 13);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_RandomShadowbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array x1,vx_array y1,vx_array x2,vx_array y2,vx_array numberOfShadows,vx_array maxSizeX,vx_array maxSizeY,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) x1,
			(vx_reference) y1,
			(vx_reference) x2,
			(vx_reference) y2,
			(vx_reference) numberOfShadows,
			(vx_reference) maxSizeX,
			(vx_reference) maxSizeY,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_RANDOMSHADOWBATCHPD, params, 13);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_RandomShadowbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array x1,vx_array y1,vx_array x2,vx_array y2,vx_array numberOfShadows,vx_array maxSizeX,vx_array maxSizeY,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) x1,
			(vx_reference) y1,
			(vx_reference) x2,
			(vx_reference) y2,
			(vx_reference) numberOfShadows,
			(vx_reference) maxSizeX,
			(vx_reference) maxSizeY,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_RANDOMSHADOWBATCHPDROID, params, 17);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Fog(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar fogValue)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) fogValue,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_FOG, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_FogbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar fogValue,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) fogValue,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_FOGBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_FogbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array fogValue,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) fogValue,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_FOGBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_FogbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array fogValue,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) fogValue,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_FOGBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Rain(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar rainValue,vx_scalar rainWidth,vx_scalar rainHeight,vx_scalar rainTransperancy)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) rainValue,
			(vx_reference) rainWidth,
			(vx_reference) rainHeight,
			(vx_reference) rainTransperancy,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_RAIN, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_RainbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar rainValue,vx_scalar rainWidth,vx_scalar rainHeight,vx_scalar rainTransperancy,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) rainValue,
			(vx_reference) rainWidth,
			(vx_reference) rainHeight,
			(vx_reference) rainTransperancy,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_RAINBATCHPS, params, 10);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_RainbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array rainValue,vx_array rainWidth,vx_array rainHeight,vx_array rainTransperancy,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) rainValue,
			(vx_reference) rainWidth,
			(vx_reference) rainHeight,
			(vx_reference) rainTransperancy,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_RAINBATCHPD, params, 10);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_RainbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array rainValue,vx_array rainWidth,vx_array rainHeight,vx_array rainTransperancy,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) rainValue,
			(vx_reference) rainWidth,
			(vx_reference) rainHeight,
			(vx_reference) rainTransperancy,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_RAINBATCHPDROID, params, 14);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_RandomCropLetterBox(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar x1,vx_scalar y1,vx_scalar x2,vx_scalar y2)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) x1,
			(vx_reference) y1,
			(vx_reference) x2,
			(vx_reference) y2,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_RANDOMCROPLETTERBOX, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_RandomCropLetterBoxbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar dstImgWidth,vx_scalar dstImgHeight,vx_scalar x1,vx_scalar y1,vx_scalar x2,vx_scalar y2,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) x1,
			(vx_reference) y1,
			(vx_reference) x2,
			(vx_reference) y2,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_RANDOMCROPLETTERBOXBATCHPS, params, 12);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_RandomCropLetterBoxbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1,vx_array x2,vx_array y2,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) x1,
			(vx_reference) y1,
			(vx_reference) x2,
			(vx_reference) y2,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_RANDOMCROPLETTERBOXBATCHPD, params, 12);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_RandomCropLetterBoxbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1,vx_array x2,vx_array y2,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) x1,
			(vx_reference) y1,
			(vx_reference) x2,
			(vx_reference) y2,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_RANDOMCROPLETTERBOXBATCHPDROID, params, 17);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Exposure(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar exposureValue)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) exposureValue,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_EXPOSURE, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ExposurebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar exposureValue,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) exposureValue,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_EXPOSUREBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ExposurebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array exposureValue,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) exposureValue,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_EXPOSUREBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ExposurebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array exposureValue,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) exposureValue,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_EXPOSUREBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_HistogramBalance(vx_graph graph,vx_image pSrc,vx_image pDst)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_HISTOGRAMBALANCE, params, 3);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_HistogramBalancebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_HISTOGRAMBALANCEBATCHPS, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_HistogramBalancebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_HISTOGRAMBALANCEBATCHPD, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_HistogramBalancebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_HISTOGRAMBALANCEBATCHPDROID, params, 10);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AbsoluteDifference(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) pDst,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ABSOLUTEDIFFERENCE, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AbsoluteDifferencebatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ABSOLUTEDIFFERENCEBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AbsoluteDifferencebatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ABSOLUTEDIFFERENCEBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AbsoluteDifferencebatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ABSOLUTEDIFFERENCEBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AccumulateWeighted(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_scalar alpha)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) alpha,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ACCUMULATEWEIGHTED, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AccumulateWeightedbatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_scalar alpha,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) alpha,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ACCUMULATEWEIGHTEDBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AccumulateWeightedbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_array alpha,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) alpha,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ACCUMULATEWEIGHTEDBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AccumulateWeightedbatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_array alpha,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) alpha,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ACCUMULATEWEIGHTEDBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Accumulate(vx_graph graph,vx_image pSrc1,vx_image pSrc2)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ACCUMULATE, params, 3);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AccumulatebatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ACCUMULATEBATCHPS, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AccumulatebatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ACCUMULATEBATCHPD, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AccumulatebatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ACCUMULATEBATCHPDROID, params, 10);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Add(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) pDst,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ADD, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AddbatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ADDBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AddbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ADDBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AddbatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ADDBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Subtract(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) pDst,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_SUBTRACT, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_SubtractbatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_SUBTRACTBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_SubtractbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_SUBTRACTBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_SubtractbatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_SUBTRACTBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Magnitude(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) pDst,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_MAGNITUDE, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MagnitudebatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_MAGNITUDEBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MagnitudebatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_MAGNITUDEBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MagnitudebatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_MAGNITUDEBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Multiply(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) pDst,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_MULTIPLY, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MultiplybatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_MULTIPLYBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MultiplybatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_MULTIPLYBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MultiplybatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_MULTIPLYBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Phase(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) pDst,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_PHASE, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_PhasebatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_PHASEBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_PhasebatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_PHASEBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_PhasebatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_PHASEBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AccumulateSquared(vx_graph graph,vx_image pSrc)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ACCUMULATESQUARED, params, 2);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AccumulateSquaredbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ACCUMULATESQUAREDBATCHPS, params, 5);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AccumulateSquaredbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ACCUMULATESQUAREDBATCHPD, params, 5);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AccumulateSquaredbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ACCUMULATESQUAREDBATCHPDROID, params, 9);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BitwiseAND(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) pDst,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BITWISEAND, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BitwiseANDbatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BITWISEANDBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BitwiseANDbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BITWISEANDBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BitwiseANDbatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BITWISEANDBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BitwiseNOT(vx_graph graph,vx_image pSrc,vx_image pDst)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BITWISENOT, params, 3);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BitwiseNOTbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BITWISENOTBATCHPS, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BitwiseNOTbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BITWISENOTBATCHPD, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BitwiseNOTbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BITWISENOTBATCHPDROID, params, 10);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ExclusiveOR(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) pDst,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_EXCLUSIVEOR, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ExclusiveORbatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_EXCLUSIVEORBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ExclusiveORbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_EXCLUSIVEORBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ExclusiveORbatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_EXCLUSIVEORBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_InclusiveOR(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) pDst,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_INCLUSIVEOR, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_InclusiveORbatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_INCLUSIVEORBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_InclusiveORbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_INCLUSIVEORBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_InclusiveORbatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_INCLUSIVEORBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Histogram(vx_graph graph,vx_image pSrc,vx_array outputHistogram,vx_scalar bins)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) outputHistogram,
			(vx_reference) bins,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_HISTOGRAM, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Thresholding(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar min,vx_scalar max)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) min,
			(vx_reference) max,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_THRESHOLDING, params, 5);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ThresholdingbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar min,vx_scalar max,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) min,
			(vx_reference) max,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_THRESHOLDINGBATCHPS, params, 8);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ThresholdingbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array min,vx_array max,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) min,
			(vx_reference) max,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_THRESHOLDINGBATCHPD, params, 8);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ThresholdingbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array min,vx_array max,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) min,
			(vx_reference) max,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_THRESHOLDINGBATCHPDROID, params, 12);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Max(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) pDst,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_MAX, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MaxbatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_MAXBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MaxbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_MAXBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MaxbatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_MAXBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Min(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) pDst,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_MIN, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MinbatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_MINBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MinbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_MINBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MinbatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_MINBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MinMaxLoc(vx_graph graph,vx_image pSrc,vx_scalar min,vx_scalar max,vx_scalar minLoc,vx_scalar maxLoc)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) min,
			(vx_reference) max,
			(vx_reference) minLoc,
			(vx_reference) maxLoc,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_MINMAXLOC, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_HistogramEqualize(vx_graph graph,vx_image pSrc,vx_image pDst)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_HISTOGRAMEQUALIZE, params, 3);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_HistogramEqualizebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_HISTOGRAMEQUALIZEBATCHPS, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_HistogramEqualizebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_HISTOGRAMEQUALIZEBATCHPD, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_HistogramEqualizebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_HISTOGRAMEQUALIZEBATCHPDROID, params, 10);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MeanStddev(vx_graph graph,vx_image pSrc,vx_scalar mean,vx_scalar stdDev)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) mean,
			(vx_reference) stdDev,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_MEANSTDDEV, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Flip(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar flipAxis)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) flipAxis,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_FLIP, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_FlipbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar flipAxis,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) flipAxis,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_FLIPBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_FlipbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array flipAxis,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) flipAxis,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_FLIPBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_FlipbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array flipAxis,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) flipAxis,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_FLIPBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Resize(vx_graph graph,vx_image pSrc,vx_image pDst)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_RESIZE, params, 3);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ResizebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar dstImgWidth,vx_scalar dstImgHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_RESIZEBATCHPS, params, 8);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ResizebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_RESIZEBATCHPD, params, 8);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ResizebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_RESIZEBATCHPDROID, params, 13);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ResizeCrop(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar x1,vx_scalar y1,vx_scalar x2,vx_scalar y2)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) x1,
			(vx_reference) y1,
			(vx_reference) x2,
			(vx_reference) y2,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_RESIZECROP, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ResizeCropbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar dstImgWidth,vx_scalar dstImgHeight,vx_scalar x1,vx_scalar y1,vx_scalar x2,vx_scalar y2,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) x1,
			(vx_reference) y1,
			(vx_reference) x2,
			(vx_reference) y2,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_RESIZECROPBATCHPS, params, 12);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ResizeCropbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1,vx_array x2,vx_array y2,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) x1,
			(vx_reference) y1,
			(vx_reference) x2,
			(vx_reference) y2,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_RESIZECROPBATCHPD, params, 12);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ResizeCropbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1,vx_array x2,vx_array y2,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) x1,
			(vx_reference) y1,
			(vx_reference) x2,
			(vx_reference) y2,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_RESIZECROPBATCHPDROID, params, 17);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Rotate(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar angle)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) angle,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ROTATE, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_RotatebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar dstImgWidth,vx_scalar dstImgHeight,vx_scalar angle,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) angle,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ROTATEBATCHPS, params, 9);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_RotatebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array angle,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) angle,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ROTATEBATCHPD, params, 9);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_RotatebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array angle,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) angle,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ROTATEBATCHPDROID, params, 13);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_WarpAffine(vx_graph graph,vx_image pSrc,vx_image pDst,vx_array affine)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) affine,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_WARPAFFINE, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_WarpAffinebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar dstImgWidth,vx_scalar dstImgHeight,vx_array affine,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) affine,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_WARPAFFINEBATCHPS, params, 9);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_WarpAffinebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array affine,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) affine,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_WARPAFFINEBATCHPD, params, 9);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_WarpAffinebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array affine,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) affine,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_WARPAFFINEBATCHPDROID, params, 13);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Fisheye(vx_graph graph,vx_image pSrc,vx_image pDst)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_FISHEYE, params, 3);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_FisheyebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_FISHEYEBATCHPS, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_FisheyebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_FISHEYEBATCHPD, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_FisheyebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_FISHEYEBATCHPDROID, params, 10);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_LensCorrection(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar strength,vx_scalar zoom)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) strength,
			(vx_reference) zoom,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_LENSCORRECTION, params, 5);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_LensCorrectionbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar strength,vx_scalar zoom,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) strength,
			(vx_reference) zoom,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_LENSCORRECTIONBATCHPS, params, 8);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_LensCorrectionbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array strength,vx_array zoom,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) strength,
			(vx_reference) zoom,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_LENSCORRECTIONBATCHPD, params, 8);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_LensCorrectionbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array strength,vx_array zoom,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) strength,
			(vx_reference) zoom,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_LENSCORRECTIONBATCHPDROID, params, 12);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Scale(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar percentage)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) percentage,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_SCALE, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ScalebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar dstImgWidth,vx_scalar dstImgHeight,vx_scalar percentage,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) percentage,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_SCALEBATCHPS, params, 9);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ScalebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array percentage,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) percentage,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_SCALEBATCHPD, params, 9);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ScalebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array percentage,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) percentage,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_SCALEBATCHPDROID, params, 13);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_WarpPerspective(vx_graph graph,vx_image pSrc,vx_image pDst,vx_array perspective)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) perspective,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_WARPPERSPECTIVE, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_WarpPerspectivebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar dstImgWidth,vx_scalar dstImgHeight,vx_array perspective,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) perspective,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_WARPPERSPECTIVEBATCHPS, params, 9);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_WarpPerspectivebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array perspective,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) perspective,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_WARPPERSPECTIVEBATCHPD, params, 9);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_WarpPerspectivebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array perspective,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) perspective,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_WARPPERSPECTIVEBATCHPDROID, params, 13);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Dilate(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar kernelSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_DILATE, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_DilatebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar kernelSize,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_DILATEBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_DilatebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_DILATEBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_DilatebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_DILATEBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Erode(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar kernelSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ERODE, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ErodebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar kernelSize,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ERODEBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ErodebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ERODEBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ErodebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_ERODEBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Hue(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar hueShift)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) hueShift,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_HUE, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_HuebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar hueShift,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) hueShift,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_HUEBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_HuebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array hueShift,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) hueShift,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_HUEBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_HuebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array hueShift,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) hueShift,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_HUEBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Saturation(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar saturationFactor)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) saturationFactor,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_SATURATION, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_SaturationbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar saturationFactor,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) saturationFactor,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_SATURATIONBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_SaturationbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array saturationFactor,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) saturationFactor,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_SATURATIONBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_SaturationbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array saturationFactor,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) saturationFactor,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_SATURATIONBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ColorTemperature(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar adjustmentValue)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) adjustmentValue,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_COLORTEMPERATURE, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ColorTemperaturebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar adjustmentValue,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) adjustmentValue,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_COLORTEMPERATUREBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ColorTemperaturebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array adjustmentValue,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) adjustmentValue,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_COLORTEMPERATUREBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ColorTemperaturebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array adjustmentValue,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) adjustmentValue,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_COLORTEMPERATUREBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Vignette(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar stdDev)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) stdDev,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_VIGNETTE, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_VignettebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar stdDev,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) stdDev,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_VIGNETTEBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_VignettebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array stdDev,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) stdDev,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_VIGNETTEBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_VignettebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array stdDev,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) stdDev,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_VIGNETTEBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ChannelExtract(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar extractChannelNumber)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) extractChannelNumber,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_CHANNELEXTRACT, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ChannelExtractbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar extractChannelNumber,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) extractChannelNumber,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_CHANNELEXTRACTBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ChannelExtractbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array extractChannelNumber,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) extractChannelNumber,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_CHANNELEXTRACTBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ChannelCombine(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pSrc3,vx_image pDst)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) pSrc3,
			(vx_reference) pDst,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_CHANNELCOMBINE, params, 5);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ChannelCombinebatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pSrc3,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pSrc3,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_CHANNELCOMBINEBATCHPS, params, 8);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ChannelCombinebatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pSrc3,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pSrc3,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_CHANNELCOMBINEBATCHPD, params, 8);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_LookUpTable(vx_graph graph,vx_image pSrc,vx_image pDst,vx_array lutPtr)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) lutPtr,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_LOOKUPTABLE, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_LookUpTablebatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array lutPtr,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) lutPtr,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_LOOKUPTABLEBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_LookUpTablebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array lutPtr,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) lutPtr,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_LOOKUPTABLEBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_LookUpTablebatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array lutPtr,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) lutPtr,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_LOOKUPTABLEBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BilateralFilter(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar kernelSize,vx_scalar sigmaI,vx_scalar sigmaS)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) sigmaI,
			(vx_reference) sigmaS,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BILATERALFILTER, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BilateralFilterbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar kernelSize,vx_scalar sigmaI,vx_scalar sigmaS,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) sigmaI,
			(vx_reference) sigmaS,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BILATERALFILTERBATCHPS, params, 9);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BilateralFilterbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_array sigmaI,vx_array sigmaS,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) sigmaI,
			(vx_reference) sigmaS,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BILATERALFILTERBATCHPD, params, 9);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BilateralFilterbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_array sigmaI,vx_array sigmaS,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) sigmaI,
			(vx_reference) sigmaS,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BILATERALFILTERBATCHPDROID, params, 13);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BoxFilter(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar kernelSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BOXFILTER, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BoxFilterbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar kernelSize,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BOXFILTERBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BoxFilterbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BOXFILTERBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BoxFilterbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_BOXFILTERBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Sobel(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar sobelType)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) sobelType,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_SOBEL, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_SobelbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar sobelType,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) sobelType,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_SOBELBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_SobelbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array sobelType,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) sobelType,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_SOBELBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_SobelbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array sobelType,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) sobelType,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_SOBELBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MedianFilter(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar kernelSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_MEDIANFILTER, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MedianFilterbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar kernelSize,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_MEDIANFILTERBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MedianFilterbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_MEDIANFILTERBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MedianFilterbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_MEDIANFILTERBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_CustomConvolution(vx_graph graph,vx_image pSrc,vx_image pDst,vx_array kernel,vx_scalar kernelWidth,vx_scalar kernelHeight)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) kernel,
			(vx_reference) kernelWidth,
			(vx_reference) kernelHeight,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_CUSTOMCONVOLUTION, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_CustomConvolutionbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernel,vx_scalar kernelWidth,vx_scalar kernelHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernel,
			(vx_reference) kernelWidth,
			(vx_reference) kernelHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_CUSTOMCONVOLUTIONBATCHPS, params, 9);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_CustomConvolutionbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernel,vx_array kernelWidth,vx_array kernelHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernel,
			(vx_reference) kernelWidth,
			(vx_reference) kernelHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_CUSTOMCONVOLUTIONBATCHPD, params, 9);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_CustomConvolutionbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernel,vx_array kernelWidth,vx_array kernelHeight,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernel,
			(vx_reference) kernelWidth,
			(vx_reference) kernelHeight,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_CUSTOMCONVOLUTIONBATCHPDROID, params, 13);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_NonMaxSupression(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar kernelSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_NONMAXSUPRESSION, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_NonMaxSupressionbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar kernelSize,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_NONMAXSUPRESSIONBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_NonMaxSupressionbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_NONMAXSUPRESSIONBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_NonMaxSupressionbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_NONMAXSUPRESSIONBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_GaussianFilter(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar stdDev,vx_scalar kernelSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) stdDev,
			(vx_reference) kernelSize,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_GAUSSIANFILTER, params, 5);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_GaussianFilterbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar stdDev,vx_scalar kernelSize,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) stdDev,
			(vx_reference) kernelSize,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_GAUSSIANFILTERBATCHPS, params, 8);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_GaussianFilterbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array stdDev,vx_array kernelSize,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) stdDev,
			(vx_reference) kernelSize,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_GAUSSIANFILTERBATCHPD, params, 8);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_GaussianFilterbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array stdDev,vx_array kernelSize,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) stdDev,
			(vx_reference) kernelSize,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_GAUSSIANFILTERBATCHPDROID, params, 12);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_NonLinearFilter(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar kernelSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_NONLINEARFILTER, params, 4);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_NonLinearFilterbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar kernelSize,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_NONLINEARFILTERBATCHPS, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_NonLinearFilterbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_NONLINEARFILTERBATCHPD, params, 7);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_NonLinearFilterbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array kernelSize,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) kernelSize,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_NONLINEARFILTERBATCHPDROID, params, 11);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_LocalBinaryPattern(vx_graph graph,vx_image pSrc,vx_image pDst)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_LOCALBINARYPATTERN, params, 3);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_LocalBinaryPatternbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_LOCALBINARYPATTERNBATCHPS, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_LocalBinaryPatternbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_LOCALBINARYPATTERNBATCHPD, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_LocalBinaryPatternbatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_LOCALBINARYPATTERNBATCHPDROID, params, 10);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_DataObjectCopy(vx_graph graph,vx_image pSrc,vx_image pDst)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_DATAOBJECTCOPY, params, 3);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_DataObjectCopybatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_DATAOBJECTCOPYBATCHPS, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_DataObjectCopybatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_DATAOBJECTCOPYBATCHPD, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_DataObjectCopybatchPDROID(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_DATAOBJECTCOPYBATCHPDROID, params, 10);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_GaussianImagePyramid(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar stdDev,vx_scalar kernelSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) stdDev,
			(vx_reference) kernelSize,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_GAUSSIANIMAGEPYRAMID, params, 5);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_GaussianImagePyramidbatchPS(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar stdDev,vx_scalar kernelSize,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) stdDev,
			(vx_reference) kernelSize,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_GAUSSIANIMAGEPYRAMIDBATCHPS, params, 8);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_GaussianImagePyramidbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array stdDev,vx_array kernelSize,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) stdDev,
			(vx_reference) kernelSize,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_GAUSSIANIMAGEPYRAMIDBATCHPD, params, 8);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_LaplacianImagePyramid(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar stdDev,vx_scalar kernelSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) stdDev,
			(vx_reference) kernelSize,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_LAPLACIANIMAGEPYRAMID, params, 5);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_CannyEdgeDetector(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar max,vx_scalar min)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) max,
			(vx_reference) min,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_CANNYEDGEDETECTOR, params, 5);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_HarrisCornerDetector(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar gaussianKernelSize,vx_scalar stdDev,vx_scalar kernelSize,vx_scalar kValue,vx_scalar threshold,vx_scalar nonMaxKernelSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) gaussianKernelSize,
			(vx_reference) stdDev,
			(vx_reference) kernelSize,
			(vx_reference) kValue,
			(vx_reference) threshold,
			(vx_reference) nonMaxKernelSize,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_HARRISCORNERDETECTOR, params, 9);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_FastCornerDetector(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar noOfPixels,vx_scalar threshold,vx_scalar nonMaxKernelSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) noOfPixels,
			(vx_reference) threshold,
			(vx_reference) nonMaxKernelSize,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_FASTCORNERDETECTOR, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ControlFlow(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_image pDst,vx_scalar type)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) pDst,
			(vx_reference) type,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_CONTROLFLOW, params, 5);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ControlFlowbatchPS(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_scalar type,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) type,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_CONTROLFLOWBATCHPS, params, 8);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ControlFlowbatchPD(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array type,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) type,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_CONTROLFLOWBATCHPD, params, 8);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ControlFlowbatchPDROID(vx_graph graph,vx_image pSrc1,vx_image pSrc2,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array type,vx_array roiX,vx_array roiY,vx_array roiWidth,vx_array roiHeight,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc1,
			(vx_reference) pSrc2,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) type,
			(vx_reference) roiX,
			(vx_reference) roiY,
			(vx_reference) roiWidth,
			(vx_reference) roiHeight,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_CONTROLFLOWBATCHPDROID, params, 12);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_remap(vx_graph graph,vx_image pSrc,vx_image pDst,vx_array rowRemap,vx_array colRemap)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) rowRemap,
			(vx_reference) colRemap,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_REMAP, params, 5);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_TensorAdd(vx_graph graph,vx_array pSrc1,vx_array pSrc2,vx_array pDst,vx_scalar tensorDimensions,vx_array tensorDimensionValues)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc1,
            (vx_reference) pSrc2,
			(vx_reference) pDst,
			(vx_reference) tensorDimensions,
			(vx_reference) tensorDimensionValues,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_TENSORADD, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_TensorSubtract(vx_graph graph,vx_array pSrc1,vx_array pSrc2,vx_array pDst,vx_scalar tensorDimensions,vx_array tensorDimensionValues)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc1,
            (vx_reference) pSrc2,
			(vx_reference) pDst,
			(vx_reference) tensorDimensions,
			(vx_reference) tensorDimensionValues,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_TENSORSUBTRACT, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_TensorMultiply(vx_graph graph,vx_array pSrc1,vx_array pSrc2,vx_array pDst,vx_scalar tensorDimensions,vx_array tensorDimensionValues)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc1,
            (vx_reference) pSrc2,
			(vx_reference) pDst,
			(vx_reference) tensorDimensions,
			(vx_reference) tensorDimensionValues,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_TENSORMULTIPLY, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_TensorMatrixMultiply(vx_graph graph,vx_array pSrc1,vx_array pSrc2,vx_array pDst,vx_array tensorDimensionValues1,vx_array tensorDimensionValues2)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc1,
            (vx_reference) pSrc2,
			(vx_reference) pDst,
			(vx_reference) tensorDimensionValues1,
			(vx_reference) tensorDimensionValues2,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_TENSORMATRIXMULTIPLY, params, 6);
	}
	return node;
}
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_TensorLookup(vx_graph graph,vx_array pSrc,vx_array pDst,vx_array lutPtr,vx_scalar tensorDimensions,vx_array tensorDimensionValues)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
            (vx_reference) pDst,
			(vx_reference) lutPtr,
			(vx_reference) tensorDimensions,
			(vx_reference) tensorDimensionValues,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_TENSORLOOKUP, params, 6);
	}
	return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ColorTwist(vx_graph graph,vx_image pSrc,vx_image pDst,vx_scalar alpha,vx_scalar beta, vx_scalar hue, vx_scalar sat)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) alpha,
			(vx_reference) beta,
			(vx_reference) hue,
			(vx_reference) sat,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_COLORTWIST, params, 7);
	}
	return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ColorTwistbatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array alpha,vx_array beta, vx_array hue, vx_array sat, vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) alpha,
			(vx_reference) beta,
			(vx_reference) hue,
			(vx_reference) sat,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_COLORTWISTBATCHPD, params, 10);
	}
	return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_CropMirrorNormalizebatchPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1,vx_array mean, vx_array std_dev, vx_array flip, vx_scalar chnShift,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) x1,
			(vx_reference) y1,
			(vx_reference) mean,
			(vx_reference) std_dev,
			(vx_reference) flip,
			(vx_reference) chnShift,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		node = createNode(graph, VX_KERNEL_RPP_CROPMIRRORNORMALIZEBATCHPD, params, 14);
	}
	return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_CropPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1,vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) x1,
			(vx_reference) y1,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		node = createNode(graph, VX_KERNEL_RPP_CROPPD, params, 10);
	}
	return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ResizeCropMirrorPD(vx_graph graph,vx_image pSrc,vx_array srcImgWidth,vx_array srcImgHeight,vx_image pDst,vx_array dstImgWidth,vx_array dstImgHeight,vx_array x1,vx_array y1,vx_array x2,vx_array y2, vx_array mirror, vx_uint32 nbatchSize)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) srcImgWidth,
			(vx_reference) srcImgHeight,
			(vx_reference) pDst,
			(vx_reference) dstImgWidth,
			(vx_reference) dstImgHeight,
			(vx_reference) x1,
			(vx_reference) x2,
			(vx_reference) y1,
			(vx_reference) y2,
			(vx_reference) mirror,
			(vx_reference) NBATCHSIZE,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_RESIZECROPMIRRORPD, params, 13);
	}
	return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Copy(vx_graph graph, vx_image pSrc, vx_image pDst)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
                (vx_reference) pSrc,
                (vx_reference) pDst,
                (vx_reference) DEV_TYPE
        };
        node = createNode(graph, VX_KERNEL_RPP_COPY, params, 3);
    }
    return node;
}

//Creating node for Pixelate effect
VX_API_CALL vx_node VX_API_CALL vxExtrppNode_Nop(vx_graph graph, vx_image pSrc, vx_image pDst)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
            vx_reference params[] = {
                (vx_reference) pSrc,
                (vx_reference) pDst,
                (vx_reference) DEV_TYPE
        };
            node = createNode(graph, VX_KERNEL_RPP_NOP, params, 3);
    }
    return node;
}


VX_API_CALL vx_node VX_API_CALL  vxExtrppNode_SequenceRearrange(vx_graph graph,vx_image pSrc,vx_image pDst, vx_array newOrder, vx_uint32 newSequenceLength, vx_uint32 sequenceLength)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NEWSEQUENCELENGTH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &newSequenceLength);
		vx_scalar SEQUENCELENGTH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &sequenceLength);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) newOrder,
			(vx_reference) NEWSEQUENCELENGTH,
			(vx_reference) SEQUENCELENGTH,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_SEQUENCEREARRANGE, params, 6);
	}
	return node;
}

// utility functions
vx_node createNode(vx_graph graph, vx_enum kernelEnum, vx_reference params[], vx_uint32 num)
{
    vx_status status = VX_SUCCESS;
    vx_node node = 0;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) != VX_SUCCESS) {
        return NULL;
    }
    vx_kernel kernel = vxGetKernelByEnum(context, kernelEnum);
    if(vxGetStatus((vx_reference)kernel) == VX_SUCCESS) {
        node = vxCreateGenericNode(graph, kernel);
        if (node) {
            vx_uint32 p = 0;
            for (p = 0; p < num; p++) {
                if (params[p]) {
                    status = vxSetParameterByIndex(node, p, params[p]);
                    if (status != VX_SUCCESS) {
                        char kernelName[VX_MAX_KERNEL_NAME];
                        vxQueryKernel(kernel, VX_KERNEL_NAME, kernelName, VX_MAX_KERNEL_NAME);
                        vxAddLogEntry((vx_reference)graph, status, "createNode: vxSetParameterByIndex(%s, %d, 0x%p) => %d\n", kernelName, p, params[p], status);
                        vxReleaseNode(&node);
                        node = 0;
                        break;
                    }
                }
            }
        }
        else {
            vxAddLogEntry((vx_reference)graph, VX_ERROR_INVALID_PARAMETERS, "createNode: failed to create node with kernel enum %d\n", kernelEnum);
            status = VX_ERROR_NO_MEMORY;
        }
        vxReleaseKernel(&kernel);
    }
    else {
        vxAddLogEntry((vx_reference)graph, VX_ERROR_INVALID_PARAMETERS, "createNode: failed to retrieve kernel enum %d\n", kernelEnum);
        status = VX_ERROR_NOT_SUPPORTED;
    }
    return node;
}

#if ENABLE_OPENCL
int getEnvironmentVariable(const char * name)
{
    const char * text = getenv(name);
    if (text) {
        return atoi(text);
    }
    return -1;
}

vx_status createGraphHandle(vx_node node, RPPCommonHandle ** pHandle)
{
    RPPCommonHandle * handle = NULL;
    STATUS_ERROR_CHECK(vxGetModuleHandle(node, OPENVX_KHR_RPP, (void **)&handle));
    if(handle) {
        handle->count++;
    }
    else {
        handle = new RPPCommonHandle;
        memset(handle, 0, sizeof(*handle));
        const char * searchEnvName = "NN_MIOPEN_SEARCH";
        int isEnvSet = getEnvironmentVariable(searchEnvName);
        if (isEnvSet > 0)
            handle->exhaustiveSearch = true;

        handle->count = 1;
        STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_OPENCL_COMMAND_QUEUE, &handle->cmdq, sizeof(handle->cmdq)));

    }
    *pHandle = handle;
    return VX_SUCCESS;
}

vx_status releaseGraphHandle(vx_node node, RPPCommonHandle * handle)
{
    handle->count--;
    if(handle->count == 0) {
        //TBD: release miopen_handle
        delete handle;
        STATUS_ERROR_CHECK(vxSetModuleHandle(node, OPENVX_KHR_RPP, NULL));
    }
    return VX_SUCCESS;
}
#endif

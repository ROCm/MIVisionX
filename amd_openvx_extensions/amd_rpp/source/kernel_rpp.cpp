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

#include "internal_publishKernels.h"
#include "vx_ext_rpp.h"

vx_uint32 getGraphAffinity(vx_graph graph)
{
    AgoTargetAffinityInfo affinity;
    vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_AMD_AFFINITY, &affinity, sizeof(affinity));
    ;
    if (affinity.device_type != AGO_TARGET_AFFINITY_GPU && affinity.device_type != AGO_TARGET_AFFINITY_CPU)
        affinity.device_type = AGO_TARGET_AFFINITY_CPU;
    // std::cerr<<"\n affinity "<<affinity.device_type;
    return affinity.device_type;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BrightnessbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array alpha, vx_array beta, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)alpha,
            (vx_reference)beta,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_BRIGHTNESSBATCHPD, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_GammaCorrectionbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array gamma, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)gamma,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_GAMMACORRECTIONBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BlendbatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array alpha, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)alpha,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_BLENDBATCHPD, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BlurbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array kernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)kernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_BLURBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ContrastbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array min, vx_array max, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)min,
            (vx_reference)max,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_CONTRASTBATCHPD, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_PixelatebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_PIXELATEBATCHPD, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_JitterbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array kernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)kernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_JITTERBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_SnowbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array snowValue, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)snowValue,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_SNOWBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_NoisebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array noiseProbability, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)noiseProbability,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_NOISEBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_RandomShadowbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array x1, vx_array y1, vx_array x2, vx_array y2, vx_array numberOfShadows, vx_array maxSizeX, vx_array maxSizeY, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)x1,
            (vx_reference)y1,
            (vx_reference)x2,
            (vx_reference)y2,
            (vx_reference)numberOfShadows,
            (vx_reference)maxSizeX,
            (vx_reference)maxSizeY,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_RANDOMSHADOWBATCHPD, params, 13);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_FogbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array fogValue, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)fogValue,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_FOGBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_RainbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array rainValue, vx_array rainWidth, vx_array rainHeight, vx_array rainTransperancy, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)rainValue,
            (vx_reference)rainWidth,
            (vx_reference)rainHeight,
            (vx_reference)rainTransperancy,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_RAINBATCHPD, params, 10);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_RandomCropLetterBoxbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_array x1, vx_array y1, vx_array x2, vx_array y2, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)dstImgWidth,
            (vx_reference)dstImgHeight,
            (vx_reference)x1,
            (vx_reference)y1,
            (vx_reference)x2,
            (vx_reference)y2,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_RANDOMCROPLETTERBOXBATCHPD, params, 12);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ExposurebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array exposureValue, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)exposureValue,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_EXPOSUREBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_HistogramBalancebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_HISTOGRAMBALANCEBATCHPD, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AbsoluteDifferencebatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_ABSOLUTEDIFFERENCEBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AccumulateWeightedbatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_array alpha, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)alpha,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_ACCUMULATEWEIGHTEDBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AccumulatebatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_ACCUMULATEBATCHPD, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AddbatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_ADDBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_SubtractbatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_SUBTRACTBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MagnitudebatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_MAGNITUDEBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MultiplybatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_MULTIPLYBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_PhasebatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_PHASEBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_AccumulateSquaredbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_ACCUMULATESQUAREDBATCHPD, params, 5);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BitwiseANDbatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_BITWISEANDBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BitwiseNOTbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_BITWISENOTBATCHPD, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ExclusiveORbatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_EXCLUSIVEORBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_InclusiveORbatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_INCLUSIVEORBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Histogram(vx_graph graph, vx_image pSrc, vx_array outputHistogram, vx_scalar bins)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)outputHistogram,
            (vx_reference)bins,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_HISTOGRAM, params, 4);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ThresholdingbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array min, vx_array max, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)min,
            (vx_reference)max,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_THRESHOLDINGBATCHPD, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MaxbatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_MAXBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MinbatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_MINBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MinMaxLoc(vx_graph graph, vx_image pSrc, vx_scalar min, vx_scalar max, vx_scalar minLoc, vx_scalar maxLoc)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)min,
            (vx_reference)max,
            (vx_reference)minLoc,
            (vx_reference)maxLoc,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_MINMAXLOC, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_HistogramEqualizebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_HISTOGRAMEQUALIZEBATCHPD, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MeanStddev(vx_graph graph, vx_image pSrc, vx_scalar mean, vx_scalar stdDev)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)mean,
            (vx_reference)stdDev,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_MEANSTDDEV, params, 4);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_FlipbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array flipAxis, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)flipAxis,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_FLIPBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ResizebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)dstImgWidth,
            (vx_reference)dstImgHeight,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_RESIZEBATCHPD, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Resizetensor(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_int32 interpolation_type, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_scalar INTERPOLATION_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &interpolation_type);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)dstImgWidth,
            (vx_reference)dstImgHeight,
            (vx_reference)INTERPOLATION_TYPE,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_RESIZETENSOR, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ResizeCropbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_array x1, vx_array y1, vx_array x2, vx_array y2, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)dstImgWidth,
            (vx_reference)dstImgHeight,
            (vx_reference)x1,
            (vx_reference)y1,
            (vx_reference)x2,
            (vx_reference)y2,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_RESIZECROPBATCHPD, params, 12);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_RotatebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_array angle, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)dstImgWidth,
            (vx_reference)dstImgHeight,
            (vx_reference)angle,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_ROTATEBATCHPD, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_WarpAffinebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_array affine, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)dstImgWidth,
            (vx_reference)dstImgHeight,
            (vx_reference)affine,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_WARPAFFINEBATCHPD, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_FisheyebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_FISHEYEBATCHPD, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_LensCorrectionbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array strength, vx_array zoom, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)strength,
            (vx_reference)zoom,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_LENSCORRECTIONBATCHPD, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ScalebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_array percentage, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)dstImgWidth,
            (vx_reference)dstImgHeight,
            (vx_reference)percentage,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_SCALEBATCHPD, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_WarpPerspectivebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_array perspective, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)dstImgWidth,
            (vx_reference)dstImgHeight,
            (vx_reference)perspective,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_WARPPERSPECTIVEBATCHPD, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_DilatebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array kernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)kernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_DILATEBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ErodebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array kernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)kernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_ERODEBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_HuebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array hueShift, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)hueShift,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_HUEBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_SaturationbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array saturationFactor, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)saturationFactor,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_SATURATIONBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ColorTemperaturebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array adjustmentValue, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)adjustmentValue,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_COLORTEMPERATUREBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_VignettebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array stdDev, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)stdDev,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_VIGNETTEBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ChannelExtractbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array extractChannelNumber, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)extractChannelNumber,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_CHANNELEXTRACTBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ChannelCombinebatchPD(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pSrc3, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pSrc3,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_CHANNELCOMBINEBATCHPD, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_LookUpTablebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array lutPtr, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)lutPtr,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_LOOKUPTABLEBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_BoxFilterbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array kernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)kernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_BOXFILTERBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_SobelbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array sobelType, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)sobelType,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_SOBELBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_MedianFilterbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array kernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)kernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_MEDIANFILTERBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_CustomConvolutionbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array kernel, vx_array kernelWidth, vx_array kernelHeight, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)kernel,
            (vx_reference)kernelWidth,
            (vx_reference)kernelHeight,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_CUSTOMCONVOLUTIONBATCHPD, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_NonMaxSupressionbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array kernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)kernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_NONMAXSUPRESSIONBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_GaussianFilterbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array stdDev, vx_array kernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)stdDev,
            (vx_reference)kernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_GAUSSIANFILTERBATCHPD, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_NonLinearFilterbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array kernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)kernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_NONLINEARFILTERBATCHPD, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_LocalBinaryPatternbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_LOCALBINARYPATTERNBATCHPD, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_DataObjectCopybatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_DATAOBJECTCOPYBATCHPD, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_GaussianImagePyramidbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array stdDev, vx_array kernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)stdDev,
            (vx_reference)kernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_GAUSSIANIMAGEPYRAMIDBATCHPD, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_LaplacianImagePyramid(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_scalar stdDev, vx_scalar kernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)stdDev,
            (vx_reference)kernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_LAPLACIANIMAGEPYRAMID, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_CannyEdgeDetector(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array max, vx_array min, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)max,
            (vx_reference)min,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_CANNYEDGEDETECTOR, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_HarrisCornerDetector(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight,vx_image pDst, vx_array gaussianKernelSize, vx_array stdDev, vx_array kernelSize, vx_array kValue, vx_array threshold, vx_array nonMaxKernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)gaussianKernelSize,
            (vx_reference)stdDev,
            (vx_reference)kernelSize,
            (vx_reference)kValue,
            (vx_reference)threshold,
            (vx_reference)nonMaxKernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_HARRISCORNERDETECTOR, params, 12);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_FastCornerDetector(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array noOfPixels, vx_array threshold, vx_array nonMaxKernelSize, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)noOfPixels,
            (vx_reference)threshold,
            (vx_reference)nonMaxKernelSize,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_FASTCORNERDETECTOR, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_remap(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array rowRemap, vx_array colRemap, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)rowRemap,
            (vx_reference)colRemap,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_REMAP, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_TensorAdd(vx_graph graph, vx_array pSrc1, vx_array pSrc2, vx_array pDst, vx_scalar tensorDimensions, vx_array tensorDimensionValues)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)pDst,
            (vx_reference)tensorDimensions,
            (vx_reference)tensorDimensionValues,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_TENSORADD, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_TensorSubtract(vx_graph graph, vx_array pSrc1, vx_array pSrc2, vx_array pDst, vx_scalar tensorDimensions, vx_array tensorDimensionValues)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)pDst,
            (vx_reference)tensorDimensions,
            (vx_reference)tensorDimensionValues,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_TENSORSUBTRACT, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_TensorMultiply(vx_graph graph, vx_array pSrc1, vx_array pSrc2, vx_array pDst, vx_scalar tensorDimensions, vx_array tensorDimensionValues)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)pDst,
            (vx_reference)tensorDimensions,
            (vx_reference)tensorDimensionValues,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_TENSORMULTIPLY, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_TensorMatrixMultiply(vx_graph graph, vx_array pSrc1, vx_array pSrc2, vx_array pDst, vx_array tensorDimensionValues1, vx_array tensorDimensionValues2)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)pDst,
            (vx_reference)tensorDimensionValues1,
            (vx_reference)tensorDimensionValues2,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_TENSORMATRIXMULTIPLY, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_TensorLookup(vx_graph graph, vx_array pSrc, vx_array pDst, vx_array lutPtr, vx_scalar tensorDimensions, vx_array tensorDimensionValues)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pDst,
            (vx_reference)lutPtr,
            (vx_reference)tensorDimensions,
            (vx_reference)tensorDimensionValues,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_TENSORLOOKUP, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ColorTwistbatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array alpha, vx_array beta, vx_array hue, vx_array sat, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)alpha,
            (vx_reference)beta,
            (vx_reference)hue,
            (vx_reference)sat,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_COLORTWISTBATCHPD, params, 10);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_CropMirrorNormalizebatchPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_array x1, vx_array y1, vx_array mean, vx_array std_dev, vx_array flip, vx_scalar chnShift, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)dstImgWidth,
            (vx_reference)dstImgHeight,
            (vx_reference)x1,
            (vx_reference)y1,
            (vx_reference)mean,
            (vx_reference)std_dev,
            (vx_reference)flip,
            (vx_reference)chnShift,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_CROPMIRRORNORMALIZEBATCHPD, params, 14);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_CropPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_array x1, vx_array y1, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)dstImgWidth,
            (vx_reference)dstImgHeight,
            (vx_reference)x1,
            (vx_reference)y1,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_CROPPD, params, 10);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ResizeCropMirrorPD(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_array x1, vx_array y1, vx_array x2, vx_array y2, vx_array mirror, vx_uint32 nbatchSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_scalar NBATCHSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &nbatchSize);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)srcImgWidth,
            (vx_reference)srcImgHeight,
            (vx_reference)pDst,
            (vx_reference)dstImgWidth,
            (vx_reference)dstImgHeight,
            (vx_reference)x1,
            (vx_reference)x2,
            (vx_reference)y1,
            (vx_reference)y2,
            (vx_reference)mirror,
            (vx_reference)NBATCHSIZE,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_RESIZECROPMIRRORPD, params, 13);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ResizeMirrorNormalizeTensor(vx_graph graph, vx_image pSrc, vx_array srcImgWidth, vx_array srcImgHeight, vx_image pDst, vx_array dstImgWidth, vx_array dstImgHeight, vx_array mean, vx_array std_dev, vx_array flip, vx_scalar chnShift, vx_uint32 nbatchSize)
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
            (vx_reference) mean,
            (vx_reference) std_dev,
            (vx_reference) flip,
            (vx_reference) chnShift,
            (vx_reference) NBATCHSIZE,
            (vx_reference) DEV_TYPE
        };
        node = createNode(graph, VX_KERNEL_RPP_RESIZEMIRRORNORMALIZETENSOR, params, 12);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Copy(vx_graph graph, vx_image pSrc, vx_image pDst)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pDst,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_COPY, params, 3);
    }
    return node;
}

//Creating node for Pixelate effect
VX_API_CALL vx_node VX_API_CALL vxExtrppNode_Nop(vx_graph graph, vx_image pSrc, vx_image pDst)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS)
    {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pDst,
            (vx_reference)DEV_TYPE};
        node = createNode(graph, VX_KERNEL_RPP_NOP, params, 3);
    }
    return node;
}

VX_API_CALL vx_node VX_API_CALL  vxExtrppNode_SequenceRearrange(vx_graph graph,vx_image pSrc,vx_image pDst, vx_array newOrder, vx_uint32 newSequenceLength, vx_uint32 sequenceLength, vx_uint32 sequenceCount)
{
	vx_node node = NULL;
	vx_context context = vxGetContext((vx_reference)graph);
	if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
		vx_uint32 dev_type = getGraphAffinity(graph);
		vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
		vx_scalar NEWSEQUENCELENGTH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &newSequenceLength);
		vx_scalar SEQUENCELENGTH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &sequenceLength);
		vx_scalar SEQUENCECOUNT = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &sequenceCount);
		vx_reference params[] = {
			(vx_reference) pSrc,
			(vx_reference) pDst,
			(vx_reference) newOrder,
			(vx_reference) NEWSEQUENCELENGTH,
			(vx_reference) SEQUENCELENGTH,
			(vx_reference) SEQUENCECOUNT,
			(vx_reference) DEV_TYPE
		};
		 node = createNode(graph, VX_KERNEL_RPP_SEQUENCEREARRANGE, params, 7);
	}
	return node;
}

// utility functions
vx_node createNode(vx_graph graph, vx_enum kernelEnum, vx_reference params[], vx_uint32 num)
{
    vx_status status = VX_SUCCESS;
    vx_node node = 0;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) != VX_SUCCESS)
    {
        return NULL;
    }
    vx_kernel kernel = vxGetKernelByEnum(context, kernelEnum);
    if (vxGetStatus((vx_reference)kernel) == VX_SUCCESS)
    {
        node = vxCreateGenericNode(graph, kernel);
        if (node)
        {
            vx_uint32 p = 0;
            for (p = 0; p < num; p++)
            {
                if (params[p])
                {
                    status = vxSetParameterByIndex(node, p, params[p]);
                    if (status != VX_SUCCESS)
                    {
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
        else
        {
            vxAddLogEntry((vx_reference)graph, VX_ERROR_INVALID_PARAMETERS, "createNode: failed to create node with kernel enum %d\n", kernelEnum);
            status = VX_ERROR_NO_MEMORY;
        }
        vxReleaseKernel(&kernel);
    }
    else
    {
        vxAddLogEntry((vx_reference)graph, VX_ERROR_INVALID_PARAMETERS, "createNode: failed to retrieve kernel enum %d\n", kernelEnum);
        status = VX_ERROR_NOT_SUPPORTED;
    }
    return node;
}

vx_status createRPPHandle(vx_node node, vxRppHandle **pHandle, Rpp32u batchSize, Rpp32u deviceType) {
    vxRppHandle *handle = NULL;
    STATUS_ERROR_CHECK(vxGetModuleHandle(node, OPENVX_KHR_RPP, (void **)&handle));
    vx_uint32 cpu_num_threads;
    STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_CPU_NUM_THREADS, &cpu_num_threads, sizeof(cpu_num_threads)));

    if (handle) {
        handle->count++;
    } else {
        handle = new vxRppHandle;
        memset(handle, 0, sizeof(*handle));
        handle->count = 1;
        
        if (deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL
            STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_OPENCL_COMMAND_QUEUE, &handle->cmdq, sizeof(handle->cmdq)));
            rppCreateWithStreamAndBatchSize(&data->rppHandle, handle->cmdq, batchSize);
#elif ENABLE_HIP
            STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_HIP_STREAM, &handle->hipstream, sizeof(handle->hipstream)));
            rppCreateWithStreamAndBatchSize(&handle->rppHandle, handle->hipstream, batchSize);
#endif
        } else if (deviceType == AGO_TARGET_AFFINITY_CPU) {
            rppCreateWithBatchSize(&handle->rppHandle, batchSize, cpu_num_threads);
        }
        
        STATUS_ERROR_CHECK(vxSetModuleHandle(node, OPENVX_KHR_RPP, handle));
    }
    *pHandle = handle;
    return VX_SUCCESS;
}

vx_status releaseRPPHandle(vx_node node, vxRppHandle *handle, Rpp32u deviceType) {
    handle->count--;
    if (handle->count == 0) {
        if(deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_OPENCL || ENABLE_HIP
            rppDestroyGPU(handle->rppHandle);
#endif   
        } else if (deviceType == AGO_TARGET_AFFINITY_CPU) {
            rppDestroyHost(handle->rppHandle);
        }

        delete handle;
        STATUS_ERROR_CHECK(vxSetModuleHandle(node, OPENVX_KHR_RPP, NULL));
    }
    return VX_SUCCESS;
}

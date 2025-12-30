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


//tensor

VX_API_ENTRY vx_node VX_API_CALL vxExtRppBrightness(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pAlpha, vx_array pBeta, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pAlpha,
            (vx_reference)pBeta,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_BRIGHTNESS, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtPythonFunction(vx_graph graph, vx_tensor pSrc, vx_tensor pDst, vx_scalar bridgeFnPtr, vx_scalar functionId, vx_scalar inputLayout, vx_scalar outputLayout) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pDst,
            (vx_reference)bridgeFnPtr,
            (vx_reference)functionId,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)deviceType
        };
        node = createNode(graph, VX_KERNEL_PYTHONFUNCTION, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppCopy(vx_graph graph, vx_tensor pSrc, vx_tensor pDst) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pDst,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_COPY, params, 3);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppCropMirrorNormalize(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pMultiplier, vx_array pOffset, vx_array pMirror, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pMultiplier,
            (vx_reference)pOffset,
            (vx_reference)pMirror,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_CROPMIRRORNORMALIZE, params, 10);
    }
    return node;
}

VX_API_CALL vx_node VX_API_CALL vxExtRppNop(vx_graph graph, vx_tensor pSrc, vx_tensor pDst) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pDst,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_NOP, params, 3);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppResize(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pDstWidth, vx_array pDstHeight, vx_scalar interpolationType, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pDstWidth,
            (vx_reference)pDstHeight,
            (vx_reference)interpolationType,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_RESIZE, params, 10);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppBlend(vx_graph graph, vx_tensor pSrc1, vx_tensor pSrc2, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pShift, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pShift,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_BLEND, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppBlur(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_BLUR, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppColorTwist(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pAlpha, vx_array pBeta, vx_array pHue, vx_array pSat, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pAlpha,
            (vx_reference)pBeta,
            (vx_reference)pHue,
            (vx_reference)pSat,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_COLORTWIST, params, 11);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppContrast(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pContrastFactor, vx_array pContrastCenter, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pContrastFactor,
            (vx_reference)pContrastCenter,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_CONTRAST, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppColorTemperature(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pAdjustValue, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pAdjustValue,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_COLORTEMPERATURE, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppCrop(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_CROP, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppExposure(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pExposureFactor, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pExposureFactor,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_EXPOSURE, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppFishEye(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_FISHEYE, params, 7);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppFlip(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pHorizontalFlag, vx_array pVerticalFlag, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pHorizontalFlag,
            (vx_reference)pVerticalFlag,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_FLIP, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppFog(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pIntensityFactor, vx_array pGrayFactor, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pIntensityFactor,
            (vx_reference)pGrayFactor,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_FOG, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppGammaCorrection(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pGamma, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pGamma,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_GAMMACORRECTION, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppGlitch(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst,vx_array pXoffsetR, vx_array pYoffsetR, vx_array pXoffsetG, vx_array pYoffsetG, vx_array pXoffsetB, vx_array pYoffsetB, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pXoffsetR,
            (vx_reference)pYoffsetR,
            (vx_reference)pXoffsetG,
            (vx_reference)pYoffsetG,
            (vx_reference)pXoffsetB,
            (vx_reference)pYoffsetB,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_GLITCH, params, 13);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppHue(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pHueShift, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pHueShift,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_HUE, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppJitter(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pKernelSize, vx_scalar seed, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph); 
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pKernelSize,
            (vx_reference)seed,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_JITTER, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppLensCorrection(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pCameraMatrix, vx_array pDistortionCoeffs, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pCameraMatrix,
            (vx_reference)pDistortionCoeffs,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_LENSCORRECTION, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppNoise(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pNoiseProb, vx_array pSaltProb, vx_array pSaltValue, vx_array pPepperValue, vx_scalar seed,vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pNoiseProb,
            (vx_reference)pSaltProb,
            (vx_reference)pSaltValue,
            (vx_reference)pPepperValue,
            (vx_reference)seed,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_NOISE, params, 12);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppPixelate(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_scalar pixelationPercentage, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pixelationPercentage,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_PIXELATE, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppRain(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_scalar rainPercentage, vx_scalar rainWidth, vx_scalar rainHeight, vx_scalar rainSlantAngle, vx_array pRainTransperancy, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)rainPercentage,
            (vx_reference)rainWidth,
            (vx_reference)rainHeight,
            (vx_reference)rainSlantAngle,
            (vx_reference)pRainTransperancy,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_RAIN, params, 12);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppResizeCrop(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pDstWidth, vx_array pDstHeight,vx_scalar interpolationType, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pDstWidth,
            (vx_reference)pDstHeight,
            (vx_reference)interpolationType,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_RESIZECROP, params, 10);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppResizeCropMirror(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pDstWidth, vx_array pDstHeight, vx_array pMirror,vx_scalar interpolationType, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pDstWidth,
            (vx_reference)pDstHeight,
            (vx_reference)pMirror,
            (vx_reference)interpolationType,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_RESIZECROPMIRROR, params, 11);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppResizeMirrorNormalize(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst,vx_array pDstWidth,vx_array pDstHeight, vx_scalar interpolationType, vx_array pMean, vx_array pStdDev, vx_array pMirror, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pDstWidth,
            (vx_reference)pDstHeight,
            (vx_reference)interpolationType,
            (vx_reference)pMean,
            (vx_reference)pStdDev,
            (vx_reference)pMirror,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_RESIZEMIRRORNORMALIZE, params, 13);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppRotate(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pAngle, vx_scalar interpolationType, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pAngle,
            (vx_reference)interpolationType,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_ROTATE, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppSaturation(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pSaturationFactor, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pSaturationFactor,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_SATURATION, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppSnow(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pBrightnessCoefficient, vx_array pSnowThreshold, vx_array pDarkMode, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pBrightnessCoefficient,
            (vx_reference)pSnowThreshold,
            (vx_reference)pDarkMode,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_SNOW, params, 10);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppVignette(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pStdDev, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pStdDev,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_VIGNETTE, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppWarpAffine(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pAffineArray, vx_scalar interpolationType, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pAffineArray,
            (vx_reference)interpolationType,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_WARPAFFINE, params, 9);
    }
    return node;
}

VX_API_CALL vx_node VX_API_CALL vxExtRppSequenceRearrange(vx_graph graph, vx_tensor pSrc, vx_tensor pDst, vx_array pNewOrder, vx_scalar layout) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pDst,
            (vx_reference)pNewOrder,
            (vx_reference)layout,
            (vx_reference)deviceType
        };
        node = createNode(graph, VX_KERNEL_RPP_SEQUENCEREARRANGE, params, 5);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppPreemphasisFilter(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_array pPreemphCoeff, vx_scalar borderType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pPreemphCoeff,
            (vx_reference)borderType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_PREEMPHASISFILTER, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppSpectrogram(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_tensor pDstRoi, vx_array windowFunction, vx_scalar centerWindows, vx_scalar reflectPadding, vx_scalar spectrogramLayout,
                                                     vx_scalar power, vx_scalar nfft, vx_scalar windowLength, vx_scalar windowStep) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devtype = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devtype);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pDstRoi,
            (vx_reference)windowFunction,
            (vx_reference)centerWindows,
            (vx_reference)reflectPadding,
            (vx_reference)spectrogramLayout,
            (vx_reference)power,
            (vx_reference)nfft,
            (vx_reference)windowLength,
            (vx_reference)windowStep,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_SPECTROGRAM, params, 13);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppNonSilentRegionDetection(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pBegin, vx_tensor pLength, vx_scalar cutOffDB, vx_scalar referencePower, vx_scalar windowLength, vx_scalar resetInterval) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pBegin,
            (vx_reference)pLength,
            (vx_reference)cutOffDB,
            (vx_reference)referencePower,
            (vx_reference)windowLength,
            (vx_reference)resetInterval,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_NONSILENTREGIONDETECTION, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppSlice(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_tensor pDstRoi, vx_tensor pAnchor, vx_tensor pShape,
                                               vx_array pFillValue, vx_scalar policy, vx_scalar inputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pDstRoi,
            (vx_reference)pAnchor,
            (vx_reference)pShape,
            (vx_reference)pFillValue,
            (vx_reference)policy,
            (vx_reference)inputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_SLICE, params, 11);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppDownmix(vx_graph graph, vx_tensor pSrc, vx_tensor pDst, vx_tensor pSrcRoi) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pDst,
            (vx_reference)pSrcRoi,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_DOWNMIX, params, 4);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppToDecibels(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_scalar cutOffDB, vx_scalar multiplier, vx_scalar referenceMagnitude, vx_scalar inputLayout, vx_scalar outputLayout) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)cutOffDB,
            (vx_reference)multiplier,
            (vx_reference)referenceMagnitude,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_TODECIBELS, params, 9);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppResample(vx_graph graph, vx_tensor pSrc, vx_tensor pDst, vx_tensor pSrcRoi, vx_tensor pDstRoi,
                                                  vx_array pInRateTensor, vx_tensor pOutRateTensor, vx_scalar quality) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devtype = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devtype);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pDst,
            (vx_reference)pSrcRoi,
            (vx_reference)pDstRoi,
            (vx_reference)pOutRateTensor,
            (vx_reference)pInRateTensor,
            (vx_reference)quality,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_RESAMPLE, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppTensorMulScalar(vx_graph graph, vx_tensor pSrc, vx_tensor pDst, vx_scalar scalarValue) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devtype = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devtype);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pDst,
            (vx_reference)scalarValue,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_TENSORMULSCALAR, params, 4);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppTensorAddTensor(vx_graph graph, vx_tensor pSrc1, vx_tensor pSrc2, vx_tensor pDst, vx_tensor pSrcRoi, vx_tensor pDstRoi) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devtype = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devtype);
        vx_reference params[] = {
            (vx_reference)pSrc1,
            (vx_reference)pSrc2,
            (vx_reference)pDst,
            (vx_reference)pSrcRoi,
            (vx_reference)pDstRoi,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_TENSORADDTENSOR, params, 6);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppNormalize(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_tensor pDstRoi,
                                                   vx_scalar axis_mask, vx_array pMean, vx_array pStddev, vx_scalar computeMeanAndStdDev,
                                                   vx_scalar scale, vx_scalar shift, vx_scalar inputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pDstRoi,
            (vx_reference)axis_mask,
            (vx_reference)pMean,
            (vx_reference)pStddev,
            (vx_reference)computeMeanAndStdDev,
            (vx_reference)scale,
            (vx_reference)shift,
            (vx_reference)inputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_NORMALIZE, params, 13);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppMelFilterBank(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_tensor pDstRoi, vx_scalar freqHigh, vx_scalar freqLow, vx_scalar melFormula,
                                                       vx_scalar nfilter, vx_scalar normalize, vx_scalar sampleRate, vx_scalar inputLayout, vx_scalar outputLayout) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devtype = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devtype);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pDstRoi,
            (vx_reference)freqHigh,
            (vx_reference)freqLow,
            (vx_reference)melFormula,
            (vx_reference)nfilter,
            (vx_reference)normalize,
            (vx_reference)sampleRate,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_MELFILTERBANK, params, 13);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppTranspose(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst,
                                                   vx_array pPerm, vx_scalar inputLayout, vx_scalar outputLayout, vx_scalar roiType) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devType = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devType);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)pPerm,
            (vx_reference)inputLayout,
            (vx_reference)outputLayout,
            (vx_reference)roiType,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_TRANSPOSE, params, 8);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtRppLog1p(vx_graph graph, vx_tensor pSrc, vx_tensor pSrcRoi, vx_tensor pDst, vx_scalar inputLayout) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 devtype = getGraphAffinity(graph);
        vx_scalar deviceType = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &devtype);
        vx_reference params[] = {
            (vx_reference)pSrc,
            (vx_reference)pSrcRoi,
            (vx_reference)pDst,
            (vx_reference)inputLayout,
            (vx_reference)deviceType};
        node = createNode(graph, VX_KERNEL_RPP_LOG1P, params, 5);
    }
    return node;
}

RpptDataType getRpptDataType(vx_enum vxDataType) {
    switch(vxDataType) {
        case vx_type_e::VX_TYPE_FLOAT32:
            return RpptDataType::F32;
        case vx_type_e::VX_TYPE_FLOAT16:
            return RpptDataType::F16;
        case vx_type_e::VX_TYPE_INT8:
            return RpptDataType::I8;
        case vx_type_e::VX_TYPE_INT16:
            return RpptDataType::I16;
        default:
            return RpptDataType::U8;
    }
}

size_t getDataTypeSize(vx_enum vxDataType) {
    switch (vxDataType) {
#if defined(AMD_FP16_SUPPORT)
        case vx_type_e::VX_TYPE_FLOAT16:
            return sizeof(vx_float16);
#endif
        case vx_type_e::VX_TYPE_FLOAT32:
            return sizeof(vx_float32);
        case vx_type_e::VX_TYPE_INT8:
            return sizeof(vx_int8);
        case vx_type_e::VX_TYPE_INT16:
            return sizeof(vx_int16);
        case vx_type_e::VX_TYPE_INT32:
            return sizeof(vx_int32);
        case vx_type_e::VX_TYPE_UINT8:
            return sizeof(vx_uint8);
        case vx_type_e::VX_TYPE_UINT32:
            return sizeof(vx_uint32);
        default:
            throw std::runtime_error("Invalid datatype.");
    }
}

void fillDescriptionPtrfromDims(RpptDescPtr &descPtr, vxTensorLayout layout, size_t *tensorDims) {
    switch(layout) {
        case vxTensorLayout::VX_NHWC: {
            descPtr->n = tensorDims[0];
            descPtr->h = tensorDims[1];
            descPtr->w = tensorDims[2];
            descPtr->c = tensorDims[3];
            descPtr->strides.nStride = descPtr->c * descPtr->w * descPtr->h;
            descPtr->strides.hStride = descPtr->c * descPtr->w;
            descPtr->strides.wStride = descPtr->c;
            descPtr->strides.cStride = 1;
            descPtr->layout = RpptLayout::NHWC;
            break; 
        }
        case vxTensorLayout::VX_NCHW: {
            descPtr->n = tensorDims[0];
            descPtr->h = tensorDims[2];
            descPtr->w = tensorDims[3];
            descPtr->c = tensorDims[1];
            descPtr->strides.nStride = descPtr->c * descPtr->w * descPtr->h;
            descPtr->strides.cStride = descPtr->w * descPtr->h;
            descPtr->strides.hStride = descPtr->w;
            descPtr->strides.wStride = 1;
            descPtr->layout = RpptLayout::NCHW;
            break;
        }
        case vxTensorLayout::VX_NFHWC: {
            descPtr->n = tensorDims[0] * tensorDims[1];
            descPtr->h = tensorDims[2];
            descPtr->w = tensorDims[3];
            descPtr->c = tensorDims[4];
            descPtr->strides.nStride = descPtr->c * descPtr->w * descPtr->h;
            descPtr->strides.hStride = descPtr->c * descPtr->w;
            descPtr->strides.wStride = descPtr->c;
            descPtr->strides.cStride = 1;
            descPtr->layout = RpptLayout::NHWC;
            break;
        }
        case vxTensorLayout::VX_NFCHW: {
            descPtr->n = tensorDims[0] * tensorDims[1];
            descPtr->h = tensorDims[3];
            descPtr->w = tensorDims[4];
            descPtr->c = tensorDims[2];
            descPtr->strides.nStride = descPtr->c * descPtr->w * descPtr->h;
            descPtr->strides.cStride = descPtr->w * descPtr->h;
            descPtr->strides.hStride = descPtr->w;
            descPtr->strides.wStride = 1;
            descPtr->layout = RpptLayout::NCHW;
            break;
        }
        default: {
            throw std::runtime_error("Invalid layout value in fillDescriptionPtrfromDims.");
        }
    }
}

void fillAudioDescriptionPtrFromDims(RpptDescPtr &descPtr, size_t *maxTensorDims, vxTensorLayout layout) {
    descPtr->n = maxTensorDims[0];
    descPtr->h = maxTensorDims[1];
    descPtr->w = maxTensorDims[2];
    descPtr->c = 1;
    descPtr->strides.nStride = descPtr->c * descPtr->w * descPtr->h;
    descPtr->strides.hStride = descPtr->c * descPtr->w;
    descPtr->strides.wStride = descPtr->c;
    descPtr->strides.cStride = 1;
    descPtr->numDims = 4;
    if(tensorLayoutMapping.find(layout) != tensorLayoutMapping.end()) {
        descPtr->layout = tensorLayoutMapping.at(layout);
    } else {
        throw std::runtime_error("Invalid layout");
    }
}

void fillGenericDescriptionPtrfromDims(RpptGenericDescPtr &genericDescPtr, vxTensorLayout layout, size_t *tensorDims) {
    if(tensorLayoutMapping.find(layout) != tensorLayoutMapping.end())
        genericDescPtr->layout = tensorLayoutMapping.at(layout);
    else
        throw std::runtime_error("Invalid layout value in fillGenericDescriptionPtrfromDims");
    switch(layout) {
        case vxTensorLayout::VX_NHWC:
        case vxTensorLayout::VX_NCHW: {
            genericDescPtr->numDims = 4;
            genericDescPtr->dims[0] = tensorDims[0];
            genericDescPtr->dims[1] = tensorDims[1];
            genericDescPtr->dims[2] = tensorDims[2];
            genericDescPtr->dims[3] = tensorDims[3];
            genericDescPtr->strides[0] = genericDescPtr->dims[1] * genericDescPtr->dims[2] * genericDescPtr->dims[3];
            genericDescPtr->strides[1] = genericDescPtr->dims[2] * genericDescPtr->dims[3];
            genericDescPtr->strides[2] = genericDescPtr->dims[3];
            genericDescPtr->strides[3] = 1;
            break;
        }
        case vxTensorLayout::VX_NCDHW:
        case vxTensorLayout::VX_NDHWC: {
            genericDescPtr->numDims = 5;
            genericDescPtr->dims[0] = tensorDims[0];
            genericDescPtr->dims[1] = tensorDims[1];
            genericDescPtr->dims[2] = tensorDims[2];
            genericDescPtr->dims[3] = tensorDims[3];
            genericDescPtr->dims[4] = tensorDims[4];

            genericDescPtr->strides[0] = genericDescPtr->dims[1] * genericDescPtr->dims[2] * genericDescPtr->dims[3] * genericDescPtr->dims[4];
            genericDescPtr->strides[1] = genericDescPtr->dims[2] * genericDescPtr->dims[3] * genericDescPtr->dims[4];
            genericDescPtr->strides[2] = genericDescPtr->dims[3] * genericDescPtr->dims[4];
            genericDescPtr->strides[3] = genericDescPtr->dims[4];
            genericDescPtr->strides[4] = 1;
            break;
        }
        case vxTensorLayout::VX_NHW:
        case vxTensorLayout::VX_NFT:
        case vxTensorLayout::VX_NTF: {
            genericDescPtr->dims[0] = tensorDims[0];
            genericDescPtr->dims[1] = tensorDims[1];
            genericDescPtr->dims[2] = tensorDims[2];
            genericDescPtr->dims[3] = 1;
            if(genericDescPtr->dims[2] == 1)
                genericDescPtr->numDims = 2;
            else
                genericDescPtr->numDims = 3;
            genericDescPtr->strides[0] = genericDescPtr->dims[1] * genericDescPtr->dims[2] * genericDescPtr->dims[3];
            genericDescPtr->strides[1] = genericDescPtr->dims[2] * genericDescPtr->dims[3];
            genericDescPtr->strides[2] = genericDescPtr->dims[3];
            break;
        }
        default: {
            throw std::runtime_error("Invalid layout value in fillGenericDescriptionPtrfromDims.");
        }
    }
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
            rppCreate(&handle->rppHandle, batchSize, 0, handle->cmdq, RppBackend::RPP_OCL_BACKEND);
#elif ENABLE_HIP
            STATUS_ERROR_CHECK(vxQueryNode(node, VX_NODE_ATTRIBUTE_AMD_HIP_STREAM, &handle->hipstream, sizeof(handle->hipstream)));
            rppCreate(&handle->rppHandle, batchSize, 0, handle->hipstream, RppBackend::RPP_HIP_BACKEND);
#endif
        } else if (deviceType == AGO_TARGET_AFFINITY_CPU) {
            rppCreate(&handle->rppHandle, batchSize, cpu_num_threads, NULL, RppBackend::RPP_HOST_BACKEND);
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
#if ENABLE_OPENCL
            rppDestroy(handle->rppHandle, RppBackend::RPP_OCL_BACKEND);
#elif ENABLE_HIP
            rppDestroy(handle->rppHandle, RppBackend::RPP_HIP_BACKEND);
#endif   
        } else if (deviceType == AGO_TARGET_AFFINITY_CPU) {
            rppDestroy(handle->rppHandle, RppBackend::RPP_HOST_BACKEND);
        }

        delete handle;
        STATUS_ERROR_CHECK(vxSetModuleHandle(node, OPENVX_KHR_RPP, NULL));
    }
    return VX_SUCCESS;
}

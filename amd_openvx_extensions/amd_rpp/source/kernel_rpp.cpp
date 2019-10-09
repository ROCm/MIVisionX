#include "kernels_rpp.h"

vx_uint32 getGraphAffinity(vx_graph graph)
{
    AgoTargetAffinityInfo affinity;
    vxQueryGraph(graph, VX_GRAPH_ATTRIBUTE_AMD_AFFINITY,&affinity, sizeof(affinity));
    if(affinity.device_type != AGO_TARGET_AFFINITY_GPU && affinity.device_type != AGO_TARGET_AFFINITY_CPU)
        affinity.device_type = AGO_TARGET_AFFINITY_CPU;

    return affinity.device_type;
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

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_brightness(vx_graph graph, vx_image pSrc, vx_image pDst, vx_float32 alpha, vx_int32 beta)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar ALPHA = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &alpha);
        vx_scalar BETA = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &beta);
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
            vx_reference params[] = {
                (vx_reference) pSrc,
                (vx_reference) pDst,
                (vx_reference) ALPHA,
                (vx_reference) BETA,
                (vx_reference) DEV_TYPE
        };
            node = createNode(graph, VX_KERNEL_BRIGHTNESS, params, 5);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_contrast(vx_graph graph, vx_image pSrc, vx_image pDst, vx_uint32 max, vx_uint32 min)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar MAX = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &max);
        vx_scalar MIN = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &min);
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
            vx_reference params[] = {
                (vx_reference) pSrc,
                (vx_reference) pDst,
                (vx_reference) MAX,
                (vx_reference) MIN,
                (vx_reference) DEV_TYPE
        };
            node = createNode(graph, VX_KERNEL_CONTRAST, params, 5);
    }
    return node;
}


VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_blur(vx_graph graph, vx_image pSrc, vx_image pDst, vx_float32 sdev)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    vx_scalar SDEV = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &sdev);
    vx_uint32 dev_type = getGraphAffinity(graph);
    vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_reference params[] = {
            (vx_reference) pSrc,
            (vx_reference) pDst,
            (vx_reference) SDEV,
            (vx_reference) DEV_TYPE
        };
            node = createNode(graph, VX_KERNEL_BLUR, params, 4);
    }
    return node;
}

VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Flip(vx_graph graph, vx_image pSrc, vx_image pDst, vx_int32 flipAxis)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar FLIPAXIS = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &flipAxis);
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference) pSrc,
            (vx_reference) pDst,
            (vx_reference) FLIPAXIS,
            (vx_reference) DEV_TYPE
        };
            node = createNode(graph, VX_KERNEL_FLIP, params, 4);
    }
    return node;
}

// Creating node for Gamma Correction
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_GammaCorrection(vx_graph graph, vx_image pSrc1, vx_image pDst, vx_float32 gamma) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    vx_scalar GAMMA = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &gamma);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference) pSrc1,
            (vx_reference) pDst,
            (vx_reference) GAMMA,
            (vx_reference) DEV_TYPE
        };
            node = createNode(graph, VX_KERNEL_RPP_GAMMACORRECTION, params, 4);
    }
    return node;
}

// Creating node for Resize
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Resize(vx_graph graph, vx_image pSrc, vx_image pDst,
                                                                           vx_int32 DestWidth, vx_int32 DestHeight) {
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar DESTWIDTH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &DestWidth);
        vx_scalar DESTHEIGHT = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &DestHeight);
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference) pSrc,
            (vx_reference) pDst,
            (vx_reference) DESTWIDTH,
            (vx_reference) DESTHEIGHT,
            (vx_reference) DEV_TYPE
        };
            node = createNode(graph, VX_KERNEL_RPP_RESIZE, params, 5);
    }
    return node;
}

// Creating node for Resize Crop
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_ResizeCrop(vx_graph graph, vx_image pSrc, vx_image pDst,
                                                                    vx_int32 DestWidth, vx_int32 DestHeight, vx_int32 x1,
                                                                    vx_int32 y1, vx_int32 x2, vx_int32 y2){
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar DESTWIDTH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &DestWidth);
        vx_scalar DESTHEIGHT = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &DestHeight);
        vx_scalar X1 = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &x1);
        vx_scalar Y1 = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &y1);
        vx_scalar X2 = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &x2);
        vx_scalar Y2 = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &y2);
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference) pSrc,
            (vx_reference) pDst,
            (vx_reference) DESTWIDTH,
            (vx_reference) DESTHEIGHT,
            (vx_reference) X1,
            (vx_reference) Y1,
            (vx_reference) X2,
            (vx_reference) Y2,
            (vx_reference) DEV_TYPE
        };
            node = createNode(graph, VX_KERNEL_RPP_RESIZE_CROP, params, 9);
    }
    return node;
}


// Creating node for Rotate
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Rotate(vx_graph graph, vx_image pSrc, vx_image pDst, vx_int32 DestWidth, vx_int32 DestHeight, vx_float32 angle)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar DESTWIDTH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &DestWidth);
        vx_scalar DESTHEIGHT = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &DestHeight);
        vx_scalar ANGLE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &angle);
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference) pSrc,
            (vx_reference) pDst,
            (vx_reference) DESTWIDTH,
            (vx_reference) DESTHEIGHT,
            (vx_reference) ANGLE,
            (vx_reference) DEV_TYPE
        };
            node = createNode(graph, VX_KERNEL_RPP_ROTATE, params, 6);
    }
}

// Creating node for warp affine using array
// VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_WarpAffine(vx_graph graph, vx_image pSrc, vx_image pDst, vx_int32 DestWidth, 
//                                                                      vx_int32 DestHeight, vx_matrix affine){
//     vx_node node = NULL;
//     vx_context context = vxGetContext((vx_reference)graph);
//     if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
//         vx_scalar DESTWIDTH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &DestWidth);
//         vx_scalar DESTHEIGHT = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &DestHeight);
//         vx_reference params[] = {
//             (vx_reference) pSrc,
//             (vx_reference) pDst,
//             (vx_reference) DESTWIDTH,
//             (vx_reference) DESTHEIGHT,
//             (vx_reference) affine,
//         };
//             node = createNode(graph, VX_KERNEL_RPP_WARP_AFFINE, params, 5);
//     }
//     return node;
// }

// Creating node for warp affine
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_WarpAffine(vx_graph graph, vx_image pSrc, vx_image pDst, 
                                                                     vx_int32 DestWidth, vx_int32 DestHeight,
                                                                     vx_float32 affineVal1, vx_float32 affineVal2,
                                                                     vx_float32 affineVal3,vx_float32 affineVal4,
                                                                     vx_float32 affineVal5,vx_float32 affineVal6){
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar DESTWIDTH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &DestWidth);
        vx_scalar DESTHEIGHT = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &DestHeight);
        vx_scalar AFFINEVAL1 = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &affineVal1);
        vx_scalar AFFINEVAL2 = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &affineVal2);
        vx_scalar AFFINEVAL3 = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &affineVal3);
        vx_scalar AFFINEVAL4 = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &affineVal4);
        vx_scalar AFFINEVAL5 = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &affineVal5);
        vx_scalar AFFINEVAL6 = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &affineVal6);
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
            (vx_reference) pSrc,
            (vx_reference) pDst,
            (vx_reference) DESTWIDTH,
            (vx_reference) DESTHEIGHT,
            (vx_reference) AFFINEVAL1,
            (vx_reference) AFFINEVAL2,
            (vx_reference) AFFINEVAL3,
            (vx_reference) AFFINEVAL4,
            (vx_reference) AFFINEVAL5,
            (vx_reference) AFFINEVAL6,
            (vx_reference) DEV_TYPE
        };
            node = createNode(graph, VX_KERNEL_RPP_WARP_AFFINE, params, 11);
    }
    return node;
}

//Creating node for blend
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Blend(vx_graph graph, vx_image pSrc1, vx_image pSrc2, vx_image pDst, vx_float32 alpha)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar ALPHA = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &alpha);
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
            vx_reference params[] = {
                (vx_reference) pSrc1,
                (vx_reference) pSrc2,
                (vx_reference) pDst,
                (vx_reference) ALPHA,
                (vx_reference) DEV_TYPE
        };
            node = createNode(graph, VX_KERNEL_RPP_BLEND, params, 5);
    }
    return node;
}

//Creating node for exposure
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Exposure(vx_graph graph, vx_image pSrc1, vx_image pDst, vx_float32 exposureValue)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar EXPOSUREVALUE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &exposureValue);
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
            vx_reference params[] = {
                (vx_reference) pSrc1,
                (vx_reference) pDst,
                (vx_reference) EXPOSUREVALUE,
                (vx_reference) DEV_TYPE
        };
            node = createNode(graph, VX_KERNEL_RPP_EXPOSURE, params, 4);
    }
    return node;
}

//Creating node for fisheye
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Fisheye(vx_graph graph, vx_image pSrc, vx_image pDst)
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

//Creating node for snow
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Snow(vx_graph graph, vx_image pSrc, vx_image pDst, vx_float32 snowValue)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar SNOWVALUE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &snowValue);
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
            vx_reference params[] = {
                (vx_reference) pSrc,
                (vx_reference) pDst,
                (vx_reference) SNOWVALUE,
                (vx_reference) DEV_TYPE
        };
            node = createNode(graph, VX_KERNEL_RPP_SNOW, params, 4);
    }
    return node;
}
//Creating node for Vignette effect
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Vignette(vx_graph graph, vx_image pSrc, vx_image pDst, vx_float32 stdDev)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar STDDEV = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &stdDev);
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
            vx_reference params[] = {
                (vx_reference) pSrc,
                (vx_reference) pDst,
                (vx_reference) STDDEV,
                (vx_reference) DEV_TYPE
        };
            node = createNode(graph, VX_KERNEL_RPP_VIGNETTE, params, 4);
    }
    return node;
}

//Creating node for LensCorrection effect
VX_API_CALL vx_node vxExtrppNode_LensCorrection(vx_graph graph, vx_image pSrc, vx_image pDst, vx_float32 strength, vx_float32 zoom)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar STRENGTH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &strength);
        vx_scalar ZOOM = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &zoom);
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
            vx_reference params[] = {
                (vx_reference) pSrc,
                (vx_reference) pDst,
                (vx_reference) STRENGTH,
                (vx_reference) ZOOM,
                (vx_reference) DEV_TYPE
        };
            node = createNode(graph, VX_KERNEL_RPP_LENSCORRECTION, params, 5);
    }
    return node;
}

//Creating node for LensCorrection effect
VX_API_CALL vx_node VX_API_CALL vxExtrppNode_Pixelate(vx_graph graph, vx_image pSrc, vx_image pDst)
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

//Creating node for Jitter effect
VX_API_CALL vx_node VX_API_CALL vxExtrppNode_Jitter(vx_graph graph, vx_image pSrc, vx_image pDst, vx_uint32 kernelSize)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar KERNELSIZE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &kernelSize);
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
        vx_reference params[] = {
                (vx_reference) pSrc,
                (vx_reference) pDst,
                (vx_reference) KERNELSIZE,
                (vx_reference)  DEV_TYPE
        };
        node = createNode(graph, VX_KERNEL_RPP_JITTER, params, 4);
    }
    return node;

}
//Creating node for Color Temperature effect
VX_API_CALL vx_node VX_API_CALL vxExtrppNode_ColorTemperature(vx_graph graph, vx_image pSrc, vx_image pDst, vx_int32 adjustmentValue){

    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar ADJUSTMENTVALUE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &adjustmentValue);
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
            vx_reference params[] = {
                (vx_reference) pSrc,
                (vx_reference) pDst,
                (vx_reference) ADJUSTMENTVALUE,
                (vx_reference) DEV_TYPE
        };
            node = createNode(graph, VX_KERNEL_RPP_COLORTEMPERATURE, params, 4);
    }
    return node;

}
//Creating node for fog
VX_API_ENTRY vx_node VX_API_CALL vxExtrppNode_Fog(vx_graph graph, vx_image pSrc, vx_image pDst, vx_float32 fogValue)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar FOGVALUE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &fogValue);
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
            vx_reference params[] = {
                (vx_reference) pSrc,
                (vx_reference) pDst,
                (vx_reference) FOGVALUE,
                (vx_reference) DEV_TYPE
        };
            node = createNode(graph, VX_KERNEL_RPP_FOG, params, 4);
    }
    return node;
}
//Creating node for Rain effect
VX_API_CALL vx_node VX_API_CALL vxExtrppNode_Rain(vx_graph graph, vx_image pSrc, vx_image pDst, vx_float32 rainValue,vx_uint32 rainWidth, vx_uint32 rainHeight, vx_float32 rainTransparency)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar RAINVALUE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &rainValue);
        vx_scalar RAINWIDTH = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &rainWidth);
        vx_scalar RAINHEIGHT = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_INT32, &rainHeight);
        vx_scalar RAINTRANSPERANCY = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &rainTransparency);
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
            vx_reference params[] = {
                (vx_reference) pSrc,
                (vx_reference) pDst,
                (vx_reference) RAINVALUE,
                (vx_reference) RAINWIDTH,
                (vx_reference) RAINHEIGHT,
                (vx_reference) RAINTRANSPERANCY,
                (vx_reference) DEV_TYPE
        };
            node = createNode(graph, VX_KERNEL_RPP_RAIN, params, 7);
    }
    return node;

}

//Creating node for SNP noise addition
VX_API_CALL vx_node VX_API_CALL vxExtrppNode_NoiseSnp(vx_graph graph, vx_image pSrc, vx_image pDst, vx_float32 noiseProbability)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if(vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_scalar NOISEPROBABILITY = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_FLOAT32, &noiseProbability);
        vx_uint32 dev_type = getGraphAffinity(graph);
        vx_scalar DEV_TYPE = vxCreateScalar(vxGetContext((vx_reference)graph), VX_TYPE_UINT32, &dev_type);
            vx_reference params[] = {
                (vx_reference) pSrc,
                (vx_reference) pDst,
                (vx_reference) NOISEPROBABILITY,
                (vx_reference) DEV_TYPE
        };
            node = createNode(graph, VX_KERNEL_RPP_NOISESNP, params, 4);
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

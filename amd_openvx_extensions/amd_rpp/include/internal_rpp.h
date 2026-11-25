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

#ifndef _INTERNAL_RPP_H_
#define _INTERNAL_RPP_H_

#include "VX/vx.h"
#include "VX/vx_compatibility.h"
#include "vx_ext_amd.h"
#include "kernels_rpp.h"

#include "rpp/rpp.h"
#include "rpp/rppdefs.h"

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<iostream>
#include<algorithm>
#include<functional>
#include<map>
#include<limits>
#include<vector>

using namespace std;

#define OPENVX_KHR_RPP   "vx_khr_rpp"
#define ERRMSG(status, format, ...) printf("ERROR: " format, __VA_ARGS__), status
#define STATUS_ERROR_CHECK(call){vx_status status = call; if(status!= VX_SUCCESS) return status;}
#define PARAM_ERROR_CHECK(call){vx_status status = call; if(status!= VX_SUCCESS) goto exit;}
#define ERROR_CHECK_OBJECT(obj)  { vx_status status = vxGetStatus((vx_reference)(obj)); if(status != VX_SUCCESS){ vxAddLogEntry((vx_reference)(obj), status, "ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return status; }}
#define CHECK_HIP_RETURN_STATUS(x)                                                                         \
    do {                                                                                               \
        int retval = (x);                                                                              \
        if (retval != 0) {                                                                             \
            fprintf(stderr, "Runtime error: %s returned %d at %s:%d", #x, retval, __FILE__, __LINE__); \
            return VX_FAILURE;                                                                         \
        }                                                                                              \
    } while (0)
#define MAX_KERNELS 500

//! Brief Common data shared across all nodes in a graph
struct vxRppHandle {
#if ENABLE_HIP
    hipStream_t hipstream;
#endif
    rppHandle_t rppHandle;
    int count;
};

enum vxTensorLayout {
    VX_NHWC = 0,
    VX_NCHW = 1,
    VX_NFHWC = 2,
    VX_NFCHW = 3,
    VX_NHW = 4,     // Audio/2D layout
    VX_NFT = 5,     // Frequency major, Used for Spectrogram/MelFilterBank
    VX_NTF = 6,      // Time major, Used for Spectrogram/MelFilterBank
    VX_NDHWC = 7,
    VX_NCDHW = 8
};

const std::map<vxTensorLayout, RpptLayout> tensorLayoutMapping = {
    {vxTensorLayout::VX_NHWC, RpptLayout::NHWC},
    {vxTensorLayout::VX_NCHW, RpptLayout::NCHW},
    {vxTensorLayout::VX_NFHWC, RpptLayout::NHWC},
    {vxTensorLayout::VX_NFCHW, RpptLayout::NCHW},
#if RPP_AUDIO
    {vxTensorLayout::VX_NHW, RpptLayout::NHW},
    {vxTensorLayout::VX_NFT, RpptLayout::NFT},
    {vxTensorLayout::VX_NTF, RpptLayout::NTF},
#endif
    {vxTensorLayout::VX_NDHWC, RpptLayout::NDHWC},
    {vxTensorLayout::VX_NCDHW, RpptLayout::NCDHW}
};

//! Brief The utility functions
vx_node createNode(vx_graph graph, vx_enum kernelEnum, vx_reference params[], vx_uint32 num);
vx_status createRPPHandle(vx_node node, vxRppHandle ** pHandle, Rpp32u batchSize, Rpp32u deviceType);
vx_status releaseRPPHandle(vx_node node, vxRppHandle * handle, Rpp32u deviceType);
void fillDescriptionPtrfromDims(RpptDescPtr &descPtr, vxTensorLayout layout, size_t *tensorDims);
void fillGenericDescriptionPtrfromDims(RpptGenericDescPtr &genericDescPtr, vxTensorLayout layout, size_t *maxTensorDims);
void fillAudioDescriptionPtrFromDims(RpptDescPtr &descPtr, size_t *maxTensorDims, vxTensorLayout layout = vxTensorLayout::VX_NHW);
RpptDataType getRpptDataType(vx_enum dataType);
size_t getDataTypeSize(vx_enum dataType);

//! Brief Node-owned buffer for per-frame copies of per-sample parameters, so the graph's tensors are never modified
static vx_status allocFrameBuffer(Rpp32u deviceType, void **ppBuffer, size_t bytes) {
    if (*ppBuffer)
        return VX_SUCCESS;
    if (deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_HIP
        CHECK_HIP_RETURN_STATUS(hipHostMalloc(ppBuffer, bytes));
#else
        return VX_ERROR_NOT_SUPPORTED;
#endif
    } else {
        *ppBuffer = new char[bytes];
    }
    return VX_SUCCESS;
}

static void freeFrameBuffer(Rpp32u deviceType, void *pBuffer) {
    if (!pBuffer)
        return;
    if (deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_HIP
        hipHostFree(pBuffer);
#endif
    } else {
        delete[] static_cast<char *>(pBuffer);
    }
}

static vx_status allocRoiFrames(Rpp32u deviceType, RpptROI **ppRoi, size_t count) {
    return allocFrameBuffer(deviceType, reinterpret_cast<void **>(ppRoi), count * sizeof(RpptROI));
}

static void freeRoiFrames(Rpp32u deviceType, RpptROI *pRoi) {
    freeFrameBuffer(deviceType, pRoi);
}

static vx_status computeTensorLength(vx_tensor tensor, Rpp32u *tensorLength) {
    if (!tensorLength)
        return VX_FAILURE;
    size_t num_dims = 0;
    vx_status status = vxQueryTensor(tensor, VX_TENSOR_NUMBER_OF_DIMS, &num_dims, sizeof(num_dims));
    if (status != VX_SUCCESS)
        return status;
    std::vector<vx_size> dims(num_dims, 0);
    status = vxQueryTensor(tensor, VX_TENSOR_DIMS, dims.data(), sizeof(vx_size) * num_dims);
    if (status != VX_SUCCESS)
        return status;
    uint64_t length = 1;
    for (vx_size dim : dims) {
        if (dim == 0)
            return VX_ERROR_INVALID_DIMENSION;
        length *= dim;
        if (length > static_cast<uint64_t>(std::numeric_limits<Rpp32u>::max()))
            return VX_ERROR_INVALID_DIMENSION;
    }
    *tensorLength = static_cast<Rpp32u>(length);
    return VX_SUCCESS;
}

//! Brief Node-owned copy of a source tensor with real lead-in bytes, for RPP builds whose filters reject offsetInBytes below 12 * (kernelSize / 2)
struct SrcLeadIn {
    void *pBuf;
    Rpp32u offsetInBytes;
};

static void freeSrcLeadIn(Rpp32u deviceType, SrcLeadIn *pLeadIn) {
    if (!pLeadIn->pBuf)
        return;
    if (deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_HIP
        (void)hipFree(pLeadIn->pBuf);
#endif
    } else {
        delete[] static_cast<Rpp8u *>(pLeadIn->pBuf);
    }
    pLeadIn->pBuf = nullptr;
}

//! Brief Calls rppCall(src) with offsetInBytes = 0; if RPP answers RPP_ERROR_LOW_OFFSET, copies the source into
//! pLeadIn->pBuf behind a 12 * (kernelSize / 2) byte lead-in (and as much slack after it) and calls again on that copy
template <typename RppCall>
static vx_status callWithSrcLeadIn(vxRppHandle *handle, Rpp32u deviceType, vx_tensor srcTensor, RppPtr_t pSrc, RpptDescPtr pSrcDesc,
                                   Rpp32u kernelSize, SrcLeadIn *pLeadIn, RppCall rppCall) {
    if (!pLeadIn->offsetInBytes) {
        pSrcDesc->offsetInBytes = 0;
        RppStatus rppStatus = rppCall(pSrc);
        if (rppStatus != RPP_ERROR_LOW_OFFSET)
            return (rppStatus == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
        pLeadIn->offsetInBytes = 12 * (kernelSize / 2);
    }
    Rpp32u length;
    vx_enum dataType;
    STATUS_ERROR_CHECK(computeTensorLength(srcTensor, &length));
    STATUS_ERROR_CHECK(vxQueryTensor(srcTensor, VX_TENSOR_DATA_TYPE, &dataType, sizeof(dataType)));
    size_t bytes = static_cast<size_t>(length) * getDataTypeSize(dataType);
    size_t padded = bytes + 2 * pLeadIn->offsetInBytes;
    if (deviceType == AGO_TARGET_AFFINITY_GPU) {
#if ENABLE_HIP
        if (!pLeadIn->pBuf)
            CHECK_HIP_RETURN_STATUS(hipMalloc(&pLeadIn->pBuf, padded));
        CHECK_HIP_RETURN_STATUS(hipMemcpyAsync(static_cast<Rpp8u *>(pLeadIn->pBuf) + pLeadIn->offsetInBytes, pSrc, bytes, hipMemcpyDeviceToDevice, handle->hipstream));
#else
        return VX_ERROR_NOT_SUPPORTED;
#endif
    } else {
        if (!pLeadIn->pBuf)
            pLeadIn->pBuf = new Rpp8u[padded]();
        memcpy(static_cast<Rpp8u *>(pLeadIn->pBuf) + pLeadIn->offsetInBytes, pSrc, bytes);
    }
    pSrcDesc->offsetInBytes = pLeadIn->offsetInBytes;
    RppStatus rppStatus = rppCall(pLeadIn->pBuf);
    pSrcDesc->offsetInBytes = 0;
    return (rppStatus == RPP_SUCCESS) ? VX_SUCCESS : VX_FAILURE;
}

class Kernellist
{
public:
    struct node {
    public:
        std::function<vx_status(vx_context)> func;
        node* next;
    };
    int count;
    Kernellist(int max) {
        top = nullptr;
        maxnum = max;
        count = 0;
    }

    vx_status ADD(std::function<vx_status(vx_context)> element)
    {
        vx_status status = VX_SUCCESS;
        if (count == maxnum) return VX_ERROR_NO_RESOURCES;
        else
        {
            node *newTop = new node;
            if (top == nullptr) {
                newTop->func = element;
                newTop->next = nullptr;
                top = newTop;
                count++;
            }
            else {
                newTop->func = element;
                newTop->next = top;
                top = newTop;
                count++;
            }
        }
        return status;
    }

    vx_status REMOVE()
    {
        vx_status status = VX_SUCCESS;
        if (top == nullptr) return VX_ERROR_NO_RESOURCES;
        else {
            node * old = top;
            top = top->next;
            count--;
            delete(old);
        }
        return status;
    }

    vx_status PUBLISH(vx_context context)
    {
        vx_status status = VX_SUCCESS;

        if (top == nullptr) {
            vxAddLogEntry((vx_reference)context, VX_ERROR_NO_RESOURCES, "PUBLISH Fail, Kernel list is empty");
            return VX_ERROR_NO_RESOURCES;
        }

        else
        {
            node * Kernel = top;
            for (int i = 0; i < count; i++) {
                STATUS_ERROR_CHECK(Kernel->func(context));
                Kernel = Kernel->next;
            }
        }
        return status;
    }

private:
    node *top;
    int maxnum;
};

static Kernellist *Kernel_List;

#endif

/*
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include "../../../amd_openvx/openvx/hipvx/hip_common_funcs.h"
#include "amd_rpp_hip_host_decls.h"

// multiplies entire input with a constant scalar value passed
__global__ void __attribute__((visibility("default"))) HipTensorMulScalar(
    const float* srcPtr, float* dstPtr, float scalarValue,
    size_t maxTensorSize) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    if (id_x >= maxTensorSize) return;
    dstPtr[id_x] = srcPtr[id_x] * scalarValue;
}

    int HipExecTensorMulScalar(hipStream_t stream, const float* srcPtr,
                            float* dstPtr, float scalarValue,
                            size_t maxTensorSize) {
    int localThreadsX = 256, localThreadsY = 1;
    int globalThreadsX = maxTensorSize, globalThreadsY = 1;
    hipLaunchKernelGGL(HipTensorMulScalar,
                        dim3(ceil((float)globalThreadsX / localThreadsX),
                            ceil((float)globalThreadsY / localThreadsY)),
                        dim3(localThreadsX, localThreadsY), 0, stream, srcPtr,
                        dstPtr, scalarValue, maxTensorSize);
    HIP_CHECK(hipStreamSynchronize(stream));
    return VX_SUCCESS;
    }

// adds tensors of size [batchsize, height, width] with [batchsize, 1]
__global__ void __attribute__((visibility("default"))) HipTensorAddTensor(
    const float* src1Ptr, const float* src2Ptr, uint2 srcStridesNH,
    float* dstPtr, RpptROI* srcROI) {
    int id_x = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    int id_y = (hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y);
    int id_z = (hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z);

    if (id_x >= srcROI[id_z].xywhROI.roiWidth ||
        id_y >= srcROI[id_z].xywhROI.roiHeight)
        return;

    uint srcIdx, dstIdx;
    srcIdx = dstIdx = (id_z * srcStridesNH.x + id_y * srcStridesNH.y + id_x);
    dstPtr[srcIdx] = src1Ptr[srcIdx] + src2Ptr[id_z];
    }

int HipExecTensorAddTensor(hipStream_t stream, const float* src1Ptr,
                        const float* src2Ptr, float* dstPtr, RpptROI* srcROI,
                        size_t* inputTensorDims) {
    int localThreadsX, localThreadsY, localThreadsZ;
    int globalThreadsX, globalThreadsY, globalThreadsZ;
    localThreadsX = 16;
    localThreadsY = 16;
    localThreadsZ = 1;
    globalThreadsX = inputTensorDims[1];
    globalThreadsY = inputTensorDims[2];
    globalThreadsZ = inputTensorDims[0];

    // update localThreadX, localThreadsY if any of the input dimension is 1
    if (globalThreadsX == 1 || globalThreadsY == 1) {
        localThreadsX = 256;
        localThreadsY = 1;
    }
    hipLaunchKernelGGL(
        HipTensorAddTensor,
        dim3(ceil((float)globalThreadsX / localThreadsX),
            ceil((float)globalThreadsY / localThreadsY),
            ceil((float)globalThreadsZ / localThreadsZ)),
        dim3(localThreadsX, localThreadsY, localThreadsZ), 0, stream, src1Ptr,
        src2Ptr,
        make_uint2(inputTensorDims[2] * inputTensorDims[1], inputTensorDims[2]),
        dstPtr, srcROI);
    HIP_CHECK(hipStreamSynchronize(stream));
    return VX_SUCCESS;
}
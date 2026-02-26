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

#ifndef AMD_RPP_HIP_HOST_DECLS_H
#define AMD_RPP_HIP_HOST_DECLS_H

#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include <VX/vx.h>
#include "rpp/rppdefs.h"

int HipExecTensorMulScalar(hipStream_t stream,
                           const float *srcPtr,
                           float *dstPtr,
                           float scalarValue,
                           size_t maxTensorSize);

int HipExecTensorAddTensor(hipStream_t stream,
                           const float *src1Ptr,
                           const float *src2Ptr,
                           float *dstPtr,
                           RpptROI *srcROI,
                           size_t *inputTensorDims);
#endif //AMD_RPP_HIP_HOST_DECLS_H

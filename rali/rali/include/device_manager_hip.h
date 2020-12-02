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

// device manager functions for HIP backend
#if ENABLE_HIP
#pragma once
#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"
#include <vx_ext_amd.h>
#include <VX/vx_types.h>

struct DeviceResourcesHip {
    hipStream_t hip_stream;
    int device_id;
    DeviceResourcesHip() { hip_stream = nullptr; device_id = -1; }
};

class DeviceManagerHip {
public:
    DeviceManagerHip(){};

    hipError_t initialize();
    
    DeviceResourcesHip resources();

    void init_hip(vx_context context);

    ~DeviceManagerHip();

private:

    DeviceResourcesHip _resources;

};

using pRaliHip = std::shared_ptr<DeviceManagerHip>;

// kernel definitions for HIP
//                HipExecCopyInt8ToNHWC(_device.resources().hipstream, (void *)img_buffer, output_tensor, dest_buf_offset, n, c, h, w, multiplier0, multiplier1, multiplier2, offset0, offset1, offset2, reverse_channels);
int HipExecCopyInt8ToNHWC
(
    hipStream_t stream,
    const void*     inp_image_u8,
    void*     output_tensor,
    unsigned int     dst_buf_offset,
    const unsigned int     n,
    const unsigned int     c,
    const unsigned int     h,
    const unsigned int     w,
    float     multiplier0,
    float     multiplier1,
    float     multiplier2,
    float     offset0,
    float     offset1,
    float     offset2,
    unsigned int reverse_channels,
    unsigned int fp16
);

int HipExecCopyInt8ToNCHW
(
    hipStream_t stream,
    void*     inp_image_u8,
    void*     output_tensor,
    unsigned int     dst_buf_offset,
    const unsigned int     n,
    const unsigned int     c,
    const unsigned int     h,
    const unsigned int     w,
    float     multiplier0,
    float     multiplier1,
    float     multiplier2,
    float     offset0,
    float     offset1,
    float     offset2,
    unsigned int reverse_channels,
    unsigned int fp16
);
#endif
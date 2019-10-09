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


#ifndef _VX_KERNELS_RPP_H_
#define _VX_KERNELS_RPP_H_

#if ENABLE_OPENCL
#include <CL/cl.h>
#endif

#define OPENVX_KHR_RPP   "vx_khr_rpp"
//////////////////////////////////////////////////////////////////////
// SHARED_PUBLIC - shared sybols for export
// STITCH_API_ENTRY - export API symbols
#if _WIN32
#define SHARED_PUBLIC extern "C" __declspec(dllexport)
#else
#define SHARED_PUBLIC extern "C" __attribute__ ((visibility ("default")))
#endif

#define STATUS_ERROR_CHECK(call){vx_status status = call; if(status!= VX_SUCCESS) return status;}
#define PARAM_ERROR_CHECK(call){vx_status status = call; if(status!= VX_SUCCESS) goto exit;}
//! \brief The macro for error checking from OpenVX object.
#define ERROR_CHECK_OBJECT(obj)  { vx_status status = vxGetStatus((vx_reference)(obj)); if(status != VX_SUCCESS){ vxAddLogEntry((vx_reference)(obj), status, "ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return status; }}

//////////////////////////////////////////////////////////////////////
// common header files
#include <VX/vx.h>
#include <vx_ext_rpp.h>
#include <vx_ext_amd.h>
#include <iostream>
#include <string.h>

#define ERRMSG(status, format, ...) printf("ERROR: " format, __VA_ARGS__), status

#define VX_LIBRARY_RPP         1

enum vx_kernel_ext_amd_rpp_e

{
    VX_KERNEL_BRIGHTNESS  = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x001,
    VX_KERNEL_CONTRAST  = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x002,
    VX_KERNEL_BLUR  = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x003,
    VX_KERNEL_FLIP  = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x004,
    VX_KERNEL_RPP_GAMMACORRECTION = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x005,
    VX_KERNEL_RPP_RESIZE = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x006,
    VX_KERNEL_RPP_RESIZE_CROP = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x007,
    VX_KERNEL_RPP_ROTATE = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x008,
    VX_KERNEL_RPP_WARP_AFFINE = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x009,
    VX_KERNEL_RPP_BLEND = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x010,
    VX_KERNEL_RPP_EXPOSURE = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x011,
    VX_KERNEL_RPP_FISHEYE = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x012,
    VX_KERNEL_RPP_SNOW = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x013,
    VX_KERNEL_RPP_VIGNETTE = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x014,
    VX_KERNEL_RPP_LENSCORRECTION = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x015,
    VX_KERNEL_RPP_PIXELATE = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x016,
    VX_KERNEL_RPP_JITTER = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x017,
    VX_KERNEL_RPP_COLORTEMPERATURE = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x018,
    VX_KERNEL_RPP_RAIN = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x019,
    VX_KERNEL_RPP_FOG = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x020,
    VX_KERNEL_RPP_NOISESNP = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x021,
    VX_KERNEL_RPP_COPY  = VX_KERNEL_BASE(VX_ID_AMD, VX_LIBRARY_RPP) + 0x0022,
};

//////////////////////////////////////////////////////////////////////
//! \brief Common data shared across all nodes in a graph
struct RPPCommonHandle {
#if ENABLE_OPENCL
    cl_command_queue cmdq;
#endif
    void* cpuHandle = NULL;
    int count;
    bool exhaustiveSearch;
};
//////////////////////////////////////////////////////////////////////
//! \brief The utility functions
vx_node createNode(vx_graph graph, vx_enum kernelEnum, vx_reference params[], vx_uint32 num);
vx_status createGraphHandle(vx_node node, RPPCommonHandle ** pHandle);
vx_status releaseGraphHandle(vx_node node, RPPCommonHandle * handle);
int getEnvironmentVariable(const char* name);

#endif //__KERNELS_H__

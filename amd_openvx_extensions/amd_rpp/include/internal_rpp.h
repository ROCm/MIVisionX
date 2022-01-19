/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#include "rpp.h"
#include "rppdefs.h"
#include "rppi.h"

#if ENABLE_OPENCL
#include <CL/cl.h>
#endif

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<iostream>
#include<algorithm>
#include<functional>

using namespace std;

#define OPENVX_KHR_RPP   "vx_khr_rpp"
#define ERRMSG(status, format, ...) printf("ERROR: " format, __VA_ARGS__), status
#define STATUS_ERROR_CHECK(call){vx_status status = call; if(status!= VX_SUCCESS) return status;}
#define PARAM_ERROR_CHECK(call){vx_status status = call; if(status!= VX_SUCCESS) goto exit;}
#define ERROR_CHECK_OBJECT(obj)  { vx_status status = vxGetStatus((vx_reference)(obj)); if(status != VX_SUCCESS){ vxAddLogEntry((vx_reference)(obj), status, "ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return status; }}
#define MAX_KERNELS 500

//! Brief Common data shared across all nodes in a graph
struct RPPCommonHandle {
#if ENABLE_OPENCL
    cl_command_queue cmdq;
#elif ENABLE_HIP
    hipStream_t hipstream;
#endif
    void* cpuHandle = NULL;
    int count;
    bool exhaustiveSearch;
};

//! Brief The utility functions
vx_node createNode(vx_graph graph, vx_enum kernelEnum, vx_reference params[], vx_uint32 num);
vx_status createGraphHandle(vx_node node, RPPCommonHandle ** pHandle);
vx_status releaseGraphHandle(vx_node node, RPPCommonHandle * handle);
int getEnvironmentVariable(const char* name);

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

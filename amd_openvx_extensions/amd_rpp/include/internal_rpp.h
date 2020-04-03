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


#ifndef _INTERNAL_RPP_H_
#define _INTERNAL_RPP_H_

#include"VX/vx.h"
#include "kernels_rpp.h"
#include <VX/vx_compatibility.h>

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<iostream>
#include<algorithm>
#include<functional>

using namespace std;

//#define STATUS_ERROR_CHECK(call){vx_status status = call; if(status!= VX_SUCCESS) return status;}
//#define PARAM_ERROR_CHECK(call){vx_status status = call; if(status!= VX_SUCCESS) goto exit;}
//! \brief The macro for error checking from OpenVX object.
//#define ERROR_CHECK_OBJECT(obj)  { vx_status status = vxGetStatus((vx_reference)(obj)); if(status != VX_SUCCESS){ vxAddLogEntry((vx_reference)(obj), status, "ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return status; }}

#define MAX_KERNELS 100

class Kernellist
{
public:
	struct node{ public: std::function<vx_status(vx_context)> func; node* next; };
	int count;
	Kernellist(int max){ top = nullptr; maxnum = max; count = 0;}

	vx_status ADD(std::function<vx_status(vx_context)> element)
	{
		vx_status status = VX_SUCCESS;
		if (count == maxnum) return VX_ERROR_NO_RESOURCES;
		else
		{
			node *newTop = new node;
			if (top == nullptr){ newTop->func = element;	newTop->next = nullptr;  top = newTop;	count++; }
			else{ newTop->func = element;	newTop->next = top; top = newTop; count++; }
		}
		return status;
	}

	vx_status REMOVE()
	{
		vx_status status = VX_SUCCESS;
		if (top == nullptr) return VX_ERROR_NO_RESOURCES;
		else{ node * old = top; top = top->next; count--; delete(old); }
		return status;
	}

	vx_status PUBLISH(vx_context context)
	{
		vx_status status = VX_SUCCESS;

		if (top == nullptr) { vxAddLogEntry((vx_reference)context, VX_ERROR_NO_RESOURCES, "PUBLISH Fail, Kernel list is empty");  return VX_ERROR_NO_RESOURCES; }

		else
		{
			node * Kernel = top;
			for (int i = 0; i < count; i++){ STATUS_ERROR_CHECK(Kernel->func(context)); Kernel = Kernel->next;}
		}
		return status;
	}

private:
	node *top; int maxnum;
};

static Kernellist *Kernel_List;

#endif

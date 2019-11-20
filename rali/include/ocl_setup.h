#pragma  once
#include <CL/cl.h>

int get_device_and_context(int devIdx, cl_context *pContext, cl_device_id *pDevice, cl_device_type clDeviceType);

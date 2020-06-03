/* 
Copyright (c) 2015 - 2020 Advanced Micro Devices, Inc. All rights reserved.
 
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

#include "ago_internal.h"
#include "ago_haf_gpu.h"

#define ENABLE_LOCAL_DEBUG_MESSAGES                       0
#define ENABLE_DEBUG_DUMP_CL_BUFFERS                      0

#if ENABLE_DEBUG_DUMP_CL_BUFFERS
static void clDumpBuffer(const char * fileNameFormat, cl_command_queue opencl_cmdq, AgoData * data)
{
	if(!data->opencl_buffer) return;
	static int dumpBufferCount = 0; dumpBufferCount++;
	char fileName[1024]; sprintf(fileName, fileNameFormat, dumpBufferCount);
	cl_mem opencl_buffer = data->opencl_buffer;
	cl_uint opencl_buffer_offset = data->opencl_buffer_offset;
	cl_uint size = (cl_uint)data->size;
	FILE * fp = fopen(fileName, "wb"); if (!fp) { printf("ERROR: unable to create: %s\n", fileName); exit(1); }
	clFinish(opencl_cmdq);
	void * p = clEnqueueMapBuffer(opencl_cmdq, opencl_buffer, CL_TRUE, CL_MAP_READ, 0, opencl_buffer_offset + size, 0, NULL, NULL, NULL);
	fwrite(p, 1, opencl_buffer_offset + size, fp);
	clEnqueueUnmapMemObject(opencl_cmdq, opencl_buffer, p, 0, NULL, NULL);
	if (data->ref.type == VX_TYPE_IMAGE) {
		printf("OK: dumped buffer %4.4s %dx%d,%d (%d+%d bytes) into %s\n", &data->u.img.format, data->u.img.width, data->u.img.height, data->u.img.stride_in_bytes, opencl_buffer_offset, size, fileName);
	}
	else {
		printf("OK: dumped buffer (%d+%d bytes) into %s\n", opencl_buffer_offset, size, fileName);
	}
	fclose(fp);
}
#endif

#if ENABLE_OPENCL
int agoGpuOclReleaseContext(AgoContext * context)
{
	if (context->opencl_cmdq) {
		cl_int status = clReleaseCommandQueue(context->opencl_cmdq);
		if (status) {
			agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: agoGpuOclReleaseContext: clReleaseCommandQueue(%p) failed (%d)\n", context->opencl_cmdq, status);
			return -1;
		}
		context->opencl_cmdq = NULL;
	}
	if (context->opencl_context && !context->opencl_context_imported) {
		cl_int status = clReleaseContext(context->opencl_context);
		if (status) {
			agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: agoGpuOclReleaseContext: clReleaseContext(%p) failed (%d)\n", context->opencl_context, status);
			return -1;
		}
	}
	context->opencl_context = NULL;
	return 0;
}

int agoGpuOclReleaseGraph(AgoGraph * graph)
{
	if (graph->opencl_cmdq) {
		cl_int status = clReleaseCommandQueue(graph->opencl_cmdq);
		if (status) {
			agoAddLogEntry(&graph->ref, VX_FAILURE, "ERROR: agoGpuOclReleaseGraph: clReleaseCommandQueue(%p) failed (%d)\n", graph->opencl_cmdq, status);
			return -1;
		}
		graph->opencl_cmdq = NULL;
	}
	return 0;
}

int agoGpuOclReleaseSuperNode(AgoSuperNode * supernode)
{
	cl_int err;
	if (supernode->opencl_kernel) {
		err = clReleaseKernel(supernode->opencl_kernel); 
		if (err) { 
			agoAddLogEntry(NULL, VX_FAILURE, "ERROR: clReleaseKernel(%p) failed(%d)\n", supernode->opencl_kernel, err);
			return -1; 
		}
	}
	if (supernode->opencl_program) {
		err = clReleaseProgram(supernode->opencl_program); 
		if (err) { 
			agoAddLogEntry(NULL, VX_FAILURE, "ERROR: clReleaseProgram(%p) failed(%d)\n", supernode->opencl_program, err);
			return -1; 
		}
	}
	if (supernode->opencl_event) {
		clReleaseEvent(supernode->opencl_event);
	}
	return 0;
}

int agoGpuOclReleaseData(AgoData * data)
{
	if (data->opencl_buffer_allocated) {
		clReleaseMemObject(data->opencl_buffer_allocated);
		data->opencl_buffer_allocated = NULL;
	}
	if (data->opencl_svm_buffer_allocated) {
		if (data->ref.context->opencl_config_flags & CONFIG_OPENCL_SVM_AS_FGS) {
			agoReleaseMemory(data->opencl_svm_buffer_allocated);
		}
		else {
			clSVMFree(data->ref.context->opencl_context, data->opencl_svm_buffer_allocated);
		}
		data->opencl_svm_buffer_allocated = NULL;
	}
	data->opencl_buffer = NULL;
	data->opencl_svm_buffer = NULL;
	data->opencl_buffer_offset = 0;
	return 0;
}

int agoGpuOclCreateContext(AgoContext * context, cl_context opencl_context)
{
	if (opencl_context) {
		// use the given OpenCL context 
		context->opencl_context_imported = true;
		context->opencl_context = opencl_context;
	}
	else {
		// get AMD platform
		cl_uint num_platforms;
		cl_int status;
		if ((status = clGetPlatformIDs(0, NULL, &num_platforms)) != CL_SUCCESS) {
			agoAddLogEntry(NULL, VX_FAILURE, "ERROR: clGetPlatformIDs(0,0,*) => %d (failed)\n", status);
			return -1;
		}
		cl_platform_id * platform_list = new cl_platform_id[num_platforms];
		if ((status = clGetPlatformIDs(num_platforms, platform_list, NULL)) != CL_SUCCESS) {
			agoAddLogEntry(NULL, VX_FAILURE, "ERROR: clGetPlatformIDs(%d,*,0) => %d (failed)\n", num_platforms, status);
			return -1;
		}
		cl_platform_id platform_id = 0;
		for (int i = 0; i < (int)num_platforms; i++) {
			char vendor[128] = { 0 };
			if ((status = clGetPlatformInfo(platform_list[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL)) != CL_SUCCESS) {
				agoAddLogEntry(NULL, VX_FAILURE, "ERROR: clGetPlatformInfo([%d],...) => %d (failed)\n", i, status);
				return -1;
			}
			if (!strcmp(vendor, "Advanced Micro Devices, Inc.")) {
				platform_id = platform_list[i];
				break;
			}
		}
		delete [] platform_list;
		if (!platform_id) {
			agoAddLogEntry(NULL, VX_FAILURE, "ERROR: Could not find a valid AMD platform\n");
			return -1;
		}
		// set context properties
		cl_context_properties ctxprop[] = {
			CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id,
			0, 0
		};
		// create context
		context->opencl_context_imported = false;
		context->opencl_context = clCreateContextFromType(ctxprop, CL_DEVICE_TYPE_GPU, NULL, NULL, &status);
		if (!context || status != CL_SUCCESS) {
			agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: clCreateContextFromType(CL_DEVICE_TYPE_GPU) => %d (failed)\n", status);
			return -1;
		}
	}
	// get the list of GPUs
	size_t size;
	cl_int status = clGetContextInfo(context->opencl_context, CL_CONTEXT_DEVICES, sizeof(context->opencl_device_list), context->opencl_device_list, &size);
	if (status != CL_SUCCESS) {
		agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: clGetContextInfo() => %d\n", status);
		return -1;
	}
	context->opencl_num_devices = (int)(size / sizeof(cl_device_id));
	// select device id
	int device_id = 0;
	if (context->attr_affinity.device_type == AGO_TARGET_AFFINITY_GPU) {
		if ((context->attr_affinity.device_info & AGO_TARGET_AFFINITY_GPU_INFO_DEVICE_MASK) < context->opencl_num_devices) {
			device_id = context->attr_affinity.device_info & AGO_TARGET_AFFINITY_GPU_INFO_DEVICE_MASK;
		}
	}
	// get device information
	char deviceVersion[256] = { 0 };
	status = clGetDeviceInfo(context->opencl_device_list[device_id], CL_DEVICE_VERSION, sizeof(deviceVersion), deviceVersion, NULL);
	if (status) { 
		agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: clGetDeviceInfo(%p,CL_DEVICE_VERSION) => %d\n", context->opencl_device_list[device_id], status);
		return -1; 
	}
	// check for OpenCL 1.2 version: force OpenCL 1.2 if environment variable AGO_OPENCL_VERSION_CHECK=1.2
	char opencl_version_check[64] = "";
	agoGetEnvironmentVariable("AGO_OPENCL_VERSION_CHECK", opencl_version_check, sizeof(opencl_version_check));
	if (deviceVersion[7] < '2' || !strcmp(opencl_version_check, "1.2")) {
		// mark that kernels have to be OpenCL 1.2 compatible
		context->opencl_config_flags |= CONFIG_OPENCL_USE_1_2;
	}
	// get device capabilities
	char deviceName[256] = { 0 };
	status = clGetDeviceInfo(context->opencl_device_list[device_id], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
	if (status) { 
		agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: clGetDeviceInfo(%p,CL_DEVICE_NAME) => %d\n", context->opencl_device_list[device_id], status);
		return -1; 
	}
	agoAddLogEntry(&context->ref, VX_SUCCESS, "OK: OpenVX using GPU device#%d (%s) [%s] [SvmCaps " VX_FMT_SIZE " %d]\n", device_id, deviceName, deviceVersion, context->opencl_svmcaps, context->opencl_config_flags);
	memset(context->opencl_extensions, 0, sizeof(context->opencl_extensions));
	status = clGetDeviceInfo(context->opencl_device_list[device_id], CL_DEVICE_EXTENSIONS, sizeof(context->opencl_extensions), context->opencl_extensions, NULL);
	if (status) { 
		agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: clGetDeviceInfo(%p,CL_DEVICE_EXTENSIONS) => %d\n", context->opencl_device_list[device_id], status);
		return -1; 
	}
	context->opencl_svmcaps = 0;
	status = clGetDeviceInfo(context->opencl_device_list[device_id], CL_DEVICE_SVM_CAPABILITIES, sizeof(context->opencl_svmcaps), &context->opencl_svmcaps, NULL);
	if (status) { 
		agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: clGetDeviceInfo(%p,CL_DEVICE_SVM_CAPABILITIES) => %d\n", context->opencl_device_list[device_id], status);
		return -1; 
	}
	// get default OpenCL build options
	strcpy(context->opencl_build_options, (context->opencl_config_flags & CONFIG_OPENCL_USE_1_2) ? "-cl-std=CL1.2" : "-cl-std=CL2.0");
	// override build options with environment variable
	agoGetEnvironmentVariable("AGO_OPENCL_BUILD_OPTIONS", context->opencl_build_options, sizeof(context->opencl_build_options));
	// override affinity device_info
	char opencl_device_info[64] = "";
	agoGetEnvironmentVariable("AGO_OPENCL_DEVICE_INFO", opencl_device_info, sizeof(opencl_device_info));
	if (opencl_device_info[0] >= '0' && opencl_device_info[0] <= '9') {
		context->attr_affinity.device_info = atoi(opencl_device_info);
	}

	// decide SVM features
	if (context->opencl_svmcaps & (CL_DEVICE_SVM_FINE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_SYSTEM)) {
		context->opencl_config_flags &= ~CONFIG_OPENCL_SVM_MASK;
		if (context->attr_affinity.device_info & AGO_TARGET_AFFINITY_GPU_INFO_SVM_MASK) {
			// set SVM flags based on device capabilities and affinity
			context->opencl_config_flags |= CONFIG_OPENCL_SVM_ENABLE;
			if (!(context->attr_affinity.device_info & AGO_TARGET_AFFINITY_GPU_INFO_SVM_NO_FGS)) {
				if (context->opencl_svmcaps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) {
					context->opencl_config_flags |= CONFIG_OPENCL_SVM_AS_FGS;
				}
			}
			if (context->attr_affinity.device_info & AGO_TARGET_AFFINITY_GPU_INFO_SVM_AS_CLMEM) {
				if (!(context->opencl_config_flags & CONFIG_OPENCL_SVM_AS_FGS)) {
					context->opencl_config_flags |= CONFIG_OPENCL_SVM_AS_CLMEM;
				}
			}
		}
		else {
			// default: TBD (SVM not enabled, for now)
			if (context->opencl_svmcaps & CL_DEVICE_SVM_FINE_GRAIN_SYSTEM) {
				// context->opencl_config_flags |= (CONFIG_OPENCL_SVM_ENABLE | CONFIG_OPENCL_SVM_AS_FGS);
			}
			else {
				// context->opencl_config_flags |= CONFIG_OPENCL_SVM_ENABLE;
			}
		}
	}
	// create command queue for buffer sync
	context->opencl_cmdq = clCreateCommandQueueWithProperties(context->opencl_context, context->opencl_device_list[device_id], NULL, &status);
	if (status) {
		agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: clCreateCommandQueueWithProperties(%p,%p,0,*) => %d\n", context->opencl_context, context->opencl_device_list[device_id], status);
		return -1;
	}

	return 0;
}

int agoGpuOclAllocBuffer(AgoData * data)
{
	// make sure buffer is valid
	if (agoDataSanityCheckAndUpdate(data)) {
		return -1;
	}
	// allocate buffer
	AgoContext * context = data->ref.context;
	if (data->ref.type == VX_TYPE_IMAGE) {
		AgoData * dataMaster = data->u.img.roiMasterImage ? data->u.img.roiMasterImage : data; // to handle image ROI
		if (!dataMaster->opencl_buffer && !dataMaster->u.img.enableUserBufferOpenCL) {
			cl_int err = CL_SUCCESS;
			dataMaster->opencl_buffer_offset = 256 + dataMaster->u.img.stride_in_bytes;
			if (!dataMaster->buffer && !dataMaster->u.img.isUniform) {
				if (context->opencl_config_flags & CONFIG_OPENCL_SVM_ENABLE) {
					if (context->opencl_config_flags & CONFIG_OPENCL_SVM_AS_FGS) {
						// allocate SVM buffer for fine grain system access
						dataMaster->opencl_svm_buffer = dataMaster->opencl_svm_buffer_allocated = (vx_uint8 *)agoAllocMemory(dataMaster->size + dataMaster->opencl_buffer_offset);
						if (!dataMaster->opencl_svm_buffer_allocated) {
							agoAddLogEntry(&dataMaster->ref, VX_FAILURE, "ERROR: agoAllocMemory(%d) => NULL\n", (int)dataMaster->size + dataMaster->opencl_buffer_offset);
							return -1;
						}
					}
					else {
						// allocate SVM buffer
						dataMaster->opencl_svm_buffer = dataMaster->opencl_svm_buffer_allocated = (vx_uint8 *)clSVMAlloc(context->opencl_context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, dataMaster->size + dataMaster->opencl_buffer_offset, 0);
						if (!dataMaster->opencl_svm_buffer_allocated) {
							agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: clSVMAlloc(%p,CL_MEM_READ_WRITE|CL_MEM_SVM_FINE_GRAIN_BUFFER,%d,0,*) => NULL\n", context->opencl_context, (int)dataMaster->size + dataMaster->opencl_buffer_offset);
							return -1;
						}
					}
				}
			}
			if (dataMaster->opencl_svm_buffer_allocated) {
				// use svm buffer as buffer(CPU)
				dataMaster->buffer = dataMaster->opencl_svm_buffer_allocated + dataMaster->opencl_buffer_offset;
				if (context->opencl_config_flags & CONFIG_OPENCL_SVM_AS_CLMEM) {
					// use svm buffer as opencl_buffer(GPU)
					dataMaster->opencl_buffer = dataMaster->opencl_buffer_allocated = clCreateBuffer(context->opencl_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, dataMaster->size + dataMaster->opencl_buffer_offset, dataMaster->opencl_svm_buffer_allocated, &err);
				}
			}
			else {
				// allocate normal opencl_buffer
				dataMaster->opencl_buffer = dataMaster->opencl_buffer_allocated = clCreateBuffer(context->opencl_context, CL_MEM_READ_WRITE, dataMaster->size + dataMaster->opencl_buffer_offset, NULL, &err);
			}
			if (err) {
				agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: clCreateBuffer(%p,CL_MEM_READ_WRITE,%d,0,*) => %d\n", context->opencl_context, (int)dataMaster->size + dataMaster->opencl_buffer_offset, err);
				return -1;
			}
			if (dataMaster->u.img.isUniform) {
				// make sure that CPU buffer is allocated
				if (!dataMaster->buffer) {
					if (agoAllocData(dataMaster)) {
						return -1;
					}
				}
				// copy the uniform image into OpenCL buffer because there won't be any commits happening to this buffer
				cl_int err = clEnqueueWriteBuffer(context->opencl_cmdq, dataMaster->opencl_buffer, CL_TRUE, dataMaster->opencl_buffer_offset, dataMaster->size, dataMaster->buffer, 0, NULL, NULL);
				if (err) { 
					agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: agoGpuOclAllocBuffer: clEnqueueWriteBuffer() => %d\n", err);
					return -1; 
				}
				dataMaster->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
			}
		}
		if (data != dataMaster) {
			// special handling for image ROI
			data->opencl_buffer = dataMaster->opencl_buffer;
			data->opencl_svm_buffer = dataMaster->opencl_svm_buffer;
			data->opencl_buffer_offset = data->u.img.rect_roi.start_y * data->u.img.stride_in_bytes +
				((data->u.img.rect_roi.start_x * (vx_uint32)data->u.img.pixel_size_in_bits) >> 3) +
				dataMaster->opencl_buffer_offset;
		}
	}
	else if (data->ref.type == VX_TYPE_ARRAY || data->ref.type == AGO_TYPE_CANNY_STACK) {
		if (!data->opencl_buffer) {
			data->opencl_buffer_offset = DATA_OPENCL_ARRAY_OFFSET; // first few bytes reserved for numitems/stacktop
			cl_int err = CL_SUCCESS;
			if (!data->buffer) {
				if (context->opencl_config_flags & CONFIG_OPENCL_SVM_ENABLE) {
					if (context->opencl_config_flags & CONFIG_OPENCL_SVM_AS_FGS) {
						// allocate SVM buffer for fine grain system access
						data->opencl_svm_buffer = data->opencl_svm_buffer_allocated = (vx_uint8 *)agoAllocMemory(data->size + data->opencl_buffer_offset);
						if (!data->opencl_svm_buffer_allocated) {
							agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: agoAllocMemory(%d) => NULL\n", (int)data->size + data->opencl_buffer_offset);
							return -1;
						}
					}
					else {
						// allocate SVM buffer
						data->opencl_svm_buffer = data->opencl_svm_buffer_allocated = (vx_uint8 *)clSVMAlloc(context->opencl_context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, data->size + data->opencl_buffer_offset, 0);
						if (!data->opencl_svm_buffer_allocated) {
							agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clSVMAlloc(%p,CL_MEM_READ_WRITE|CL_MEM_SVM_FINE_GRAIN_BUFFER,%d,0,*) => NULL\n", context->opencl_context, (int)data->size + data->opencl_buffer_offset);
							return -1;
						}
					}
					// initialize array header which containts numitems
					if (data->opencl_svm_buffer)
						memset(data->opencl_svm_buffer, 0, data->opencl_buffer_offset);
				}
			}
			if (data->opencl_svm_buffer_allocated) {
				// use svm buffer as buffer(CPU)
				data->buffer = data->opencl_svm_buffer_allocated + data->opencl_buffer_offset;
				if (context->opencl_config_flags & CONFIG_OPENCL_SVM_AS_CLMEM) {
					// use svm buffer as opencl_buffer(GPU)
					data->opencl_buffer = data->opencl_buffer_allocated = clCreateBuffer(context->opencl_context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, data->size + data->opencl_buffer_offset, data->opencl_svm_buffer_allocated, &err);
				}
			}
			else {
				// normal opencl_buffer allocation
				data->opencl_buffer = data->opencl_buffer_allocated = clCreateBuffer(context->opencl_context, CL_MEM_READ_WRITE, data->size + data->opencl_buffer_offset, NULL, &err);
				if (data->opencl_buffer) {
					// initialize array header which containts numitems
					vx_uint32 zero = 0;
					cl_event ev = nullptr;
					err = clEnqueueFillBuffer(context->opencl_cmdq, data->opencl_buffer, &zero, sizeof(zero), 0, data->opencl_buffer_offset, 0, NULL, &ev);
					if (!err) err = clWaitForEvents(1, &ev);
				}
			}
			if (err) {
				agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: clCreateBuffer(%p,CL_MEM_READ_WRITE,%d,0,*) => %d (array/cannystack)\n", context->opencl_context, (int)data->size, err);
				return -1;
			}
		}
	}
	else if (data->ref.type == VX_TYPE_SCALAR || data->ref.type == VX_TYPE_THRESHOLD || data->ref.type == VX_TYPE_MATRIX || data->ref.type == VX_TYPE_CONVOLUTION) {
		// nothing to do
	}
	else if (data->ref.type == VX_TYPE_LUT) {
		if (!data->opencl_buffer) {
			cl_int err = -1;
			cl_image_format format = { CL_INTENSITY, CL_UNORM_INT8 };
			cl_image_desc desc = { CL_MEM_OBJECT_IMAGE1D, 256, 0, 0, 1, 0, 0, 0, 0, NULL };
			data->opencl_buffer = data->opencl_buffer_allocated = clCreateImage(context->opencl_context, CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
			if (err) {
				agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: clCreateImage(%p,CL_MEM_READ_WRITE,1D/U8,256,0,*) => %d\n", context->opencl_context, err);
				return -1;
			}
			data->opencl_buffer_offset = 0;
		}
	}
	else if (data->ref.type == VX_TYPE_REMAP) {
		if (!data->opencl_buffer) {
			cl_int err = -1;
			data->opencl_buffer = data->opencl_buffer_allocated = clCreateBuffer(context->opencl_context, CL_MEM_READ_WRITE, data->size, NULL, &err);
			if (err) {
				agoAddLogEntry(&context->ref, VX_FAILURE, "ERROR: clCreateBuffer(%p,CL_MEM_READ_WRITE,%d,0,*) => %d\n", context->opencl_context, (int)data->size, err);
				return -1;
			}
			data->opencl_buffer_offset = 0;
		}
	}
	else if (data->numChildren > 0) {
		for (vx_uint32 child = 0; child < data->numChildren; child++) {
			if (agoGpuOclAllocBuffer(data->children[child]) < 0) {
				return -1;
			}
		}
	}
	else {
		agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: agoGpuOclAllocBuffer: doesn't support object type %s of %s\n", agoEnum2Name(data->ref.type), data->name.length() ? "?" : data->name.c_str());
		return -1;
	}
	// allocate CPU buffer
	if (agoAllocData(data)) {
		return -1;
	}
	return 0;
}

int agoGpuOclAllocBuffers(AgoGraph * graph, AgoNode * node)
{
	for (vx_uint32 i = 0; i < node->paramCount; i++) {
		AgoData * data = node->paramList[i];
		if (data && !data->opencl_buffer) {
			if (agoIsPartOfDelay(data)) {
				int siblingTrace[AGO_MAX_DEPTH_FROM_DELAY_OBJECT], siblingTraceCount = 0;
				data = agoGetSiblingTraceToDelayForUpdate(data, siblingTrace, siblingTraceCount);
				if (!data) return -1;
			}
			if (agoGpuOclAllocBuffer(data) < 0) {
				return -1;
			}
		}
	}
	return 0;
}

int agoGpuOclSuperNodeMerge(AgoGraph * graph, AgoSuperNode * supernode, AgoNode * node)
{
	// sanity check
	if (!node->akernel->func && !node->akernel->opencl_codegen_callback_f) {
		agoAddLogEntry(&node->akernel->ref, VX_FAILURE, "ERROR: agoGpuOclSuperNodeMerge: doesn't support kernel %s\n", node->akernel->name);
		return -1;
	}
	// merge node into supernode
	supernode->nodeList.push_back(node);
	for (vx_uint32 i = 0; i < node->paramCount; i++) {
		AgoData * data = node->paramList[i];
		if (data) {
			size_t index = std::find(supernode->dataList.begin(), supernode->dataList.end(), data) - supernode->dataList.begin();
			if (index == supernode->dataList.size()) {
				// add data with zero entries into the lists
				AgoSuperNodeDataInfo info = { 0 };
				info.needed_as_a_kernel_argument = true;
				supernode->dataInfo.push_back(info);
				supernode->dataList.push_back(data);
				supernode->dataListForAgeDelay.push_back(data);
			}
			// update count for data direction
			supernode->dataInfo[index].argument_usage[node->parameters[i].direction]++;
		}
	}
	return 0;
}

static const char * agoGpuGetKernelFunctionName(AgoNode * node)
{
	const char * kname = node->akernel->name;
	for (const char * p = kname; *p; p++)
		if (*p == '.')
			kname = p + 1;
	return kname;
}

static const char * agoGpuImageFormat2RegType(vx_df_image format)
{
	const char * reg_type = "?";
	if (format == VX_DF_IMAGE_U1_AMD) reg_type = "U1";
	else if (format == VX_DF_IMAGE_U8) reg_type = "U8";
	else if (format == VX_DF_IMAGE_S16) reg_type = "S16";
	else if (format == VX_DF_IMAGE_U16) reg_type = "U16";
	else if (format == VX_DF_IMAGE_U32) reg_type = "U32";
	else if (format == VX_DF_IMAGE_RGB) reg_type = "U24";
	else if (format == VX_DF_IMAGE_RGBX) reg_type = "U32";
	else if (format == VX_DF_IMAGE_UYVY) reg_type = "U16";
	else if (format == VX_DF_IMAGE_YUYV) reg_type = "U16";
	else if (format == VX_DF_IMAGE_F32_AMD) reg_type = "F32";
	return reg_type;
}

int agoGpuOclDataSetBufferAsKernelArg(AgoData * data, cl_kernel opencl_kernel, vx_uint32 kernelArgIndex, vx_uint32 group)
{
	cl_int err = CL_INVALID_MEM_OBJECT;
	if (data->opencl_buffer) {
		err = clSetKernelArg(opencl_kernel, (cl_uint)kernelArgIndex, sizeof(data->opencl_buffer), &data->opencl_buffer);
		if (err) {
			agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clSetKernelArg(supernode,%d,*,buffer) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex, err, group);
			return -1;
		}
	}
	else if (data->opencl_svm_buffer) {
		err = clSetKernelArgSVMPointer(opencl_kernel, (cl_uint)kernelArgIndex, data->opencl_svm_buffer);
		if (err) {
			agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clSetKernelArgSVMPointer(supernode,%d,*,buffer) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex, err, group);
			return -1;
		}
	}
	return err;
}

static int agoGpuOclSetKernelArgs(cl_kernel opencl_kernel, vx_uint32& kernelArgIndex, AgoData * data, bool need_access, vx_uint32 dataFlags, vx_uint32 group)
{
	cl_int err;
	if (data->ref.type == VX_TYPE_IMAGE) {
		if (need_access) { // only use image objects that need read/write access
			if (dataFlags & NODE_OPENCL_TYPE_NEED_IMGSIZE) {
				err = clSetKernelArg(opencl_kernel, (cl_uint)kernelArgIndex, sizeof(data->u.img.width), &data->u.img.width);
				if (err) { 
					agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clSetKernelArg(supernode,%d,*,width) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex, err, group);
					return -1; 
				}
				kernelArgIndex++;
				err = clSetKernelArg(opencl_kernel, (cl_uint)kernelArgIndex, sizeof(data->u.img.height), &data->u.img.height);
				if (err) { 
					agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clSetKernelArg(supernode,%d,*,height) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex, err, group);
					return -1; 
				}
				kernelArgIndex++;
			}
			if (agoGpuOclDataSetBufferAsKernelArg(data, opencl_kernel, kernelArgIndex, group) < 0)
				return -1;
			kernelArgIndex++;
			err = clSetKernelArg(opencl_kernel, (cl_uint)kernelArgIndex, sizeof(data->u.img.stride_in_bytes), &data->u.img.stride_in_bytes);
			if (err) { 
				agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clSetKernelArg(supernode,%d,*,stride) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex, err, group);
				return -1; 
			}
			kernelArgIndex++;
			err = clSetKernelArg(opencl_kernel, (cl_uint)kernelArgIndex, sizeof(data->opencl_buffer_offset), &data->opencl_buffer_offset);
			if (err) { 
				agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clSetKernelArg(supernode,%d,*,offset) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex, err, group);
				return -1; 
			}
			kernelArgIndex++;
		}
	}
	else if (data->ref.type == VX_TYPE_ARRAY) {
		if (agoGpuOclDataSetBufferAsKernelArg(data, opencl_kernel, kernelArgIndex, group) < 0)
			return -1;
		kernelArgIndex++;
		err = clSetKernelArg(opencl_kernel, (cl_uint)kernelArgIndex, sizeof(data->opencl_buffer_offset), &data->opencl_buffer_offset);
		if (err) { 
			agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clSetKernelArg(supernode,%d,*,arr:offset) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex, err, group);
			return -1; 
		}
		kernelArgIndex++;
		// NOTE: capacity is used when array is atomic output and numitems is used otherwise
		err = clSetKernelArg(opencl_kernel, (cl_uint)kernelArgIndex, sizeof(vx_uint32), &data->u.arr.capacity);
		if (err) { 
			agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clSetKernelArg(supernode,%d,*,arr:capacity) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex, err, group);
			return -1; 
		}
		kernelArgIndex++;
	}
	else if (data->ref.type == AGO_TYPE_CANNY_STACK) {
		if (agoGpuOclDataSetBufferAsKernelArg(data, opencl_kernel, kernelArgIndex, group) < 0)
			return -1;
		kernelArgIndex++;
		err = clSetKernelArg(opencl_kernel, (cl_uint)kernelArgIndex, sizeof(data->opencl_buffer_offset), &data->opencl_buffer_offset);
		if (err) { 
			agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clSetKernelArg(supernode,%d,*,cannystack:offset) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex, err, group);
			return -1; 
		}
		kernelArgIndex++;
		// NOTE: count is used when cannystack is output and stacktop is used when cannystack is input
		err = clSetKernelArg(opencl_kernel, (cl_uint)kernelArgIndex, sizeof(vx_uint32), &data->u.cannystack.count);
		if (err) { 
			agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clSetKernelArg(supernode,%d,*,cannystack:count) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex, err, group);
			return -1; 
		}
		kernelArgIndex++;
	}
	else if (data->ref.type == VX_TYPE_THRESHOLD) {
		size_t size = sizeof(cl_uint);
		cl_uint2 value;
		value.s0 = data->u.thr.threshold_lower;
		if (data->u.thr.thresh_type == VX_THRESHOLD_TYPE_RANGE) {
			size = sizeof(cl_uint2);
			value.s1 = data->u.thr.threshold_upper;
		}
		err = clSetKernelArg(opencl_kernel, (cl_uint)kernelArgIndex, size, &value);
		if (err) { 
			agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clSetKernelArg(supernode,%d,%d,threshold) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex, (int)size, err, group);
			return -1; 
		}
		kernelArgIndex++;
	}
	else if (data->ref.type == VX_TYPE_SCALAR) {
		err = clSetKernelArg(opencl_kernel, (cl_uint)kernelArgIndex, sizeof(cl_uint), &data->u.scalar.u.u);
		if (err) { 
			agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clSetKernelArg(supernode,%d,*,scalar) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex, err, group);
			return -1; 
		}
		kernelArgIndex++;
	}
	else if (data->ref.type == VX_TYPE_MATRIX) {
		err = clSetKernelArg(opencl_kernel, (cl_uint)kernelArgIndex, data->size, data->buffer);
		if (err) { 
			agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clSetKernelArg(supernode,%d,*,matrix) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex, err, group);
			return -1; 
		}
		kernelArgIndex++;
	}
	else if (data->ref.type == VX_TYPE_CONVOLUTION) {
		agoAllocData(data); // make sure that the data has been allocated
		err = clSetKernelArg(opencl_kernel, (cl_uint)kernelArgIndex, data->size << 1, data->reserved);
		if (err) { 
			agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clSetKernelArg(supernode,%d,*,convolution) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex, err, group);
			return -1; 
		}
		kernelArgIndex++;
	}
	else if (data->ref.type == VX_TYPE_LUT) {
		if (agoGpuOclDataSetBufferAsKernelArg(data, opencl_kernel, kernelArgIndex, group) < 0)
			return -1;
		kernelArgIndex++;
	}
	else if (data->ref.type == VX_TYPE_REMAP) {
		if (agoGpuOclDataSetBufferAsKernelArg(data, opencl_kernel, kernelArgIndex, group) < 0)
			return -1;
		kernelArgIndex++;
		vx_uint32 stride = data->u.remap.dst_width * sizeof(vx_uint32);
		err = clSetKernelArg(opencl_kernel, (cl_uint)kernelArgIndex, sizeof(stride), &stride);
		if (err) { 
			agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clSetKernelArg(supernode,%d,*,stride) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex, err, group);
			return -1; 
		}
		kernelArgIndex++;
	}
	else if (data->ref.type == VX_TYPE_SCALAR) {
		err = clSetKernelArg(opencl_kernel, (cl_uint)kernelArgIndex, sizeof(data->u.scalar.u.i), &data->u.scalar.u.i);
		if (err) { 
			agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clSetKernelArg(supernode,%d,*,scalar) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex, err, group);
			return -1; 
		}
		kernelArgIndex++;
	}
	else {
		agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: agoGpuOclSetKernelArgs: doesn't support object type %s in group#%d for kernel arg setting\n", agoEnum2Name(data->ref.type), group);
		return -1;
	}
	return 0;
}

static int agoGpuOclDataInputSync(AgoGraph * graph, cl_kernel opencl_kernel, vx_uint32& kernelArgIndex, AgoData * data, vx_uint32 dataFlags, vx_uint32 group, bool need_access, bool need_read_access, bool need_atomic_access)
{
	cl_command_queue opencl_cmdq = graph->opencl_cmdq ? graph->opencl_cmdq : graph->ref.context->opencl_cmdq;
	cl_int err;
	if (data->ref.type == VX_TYPE_IMAGE) {
		if (need_access) { // only use image objects that need read access
			if (dataFlags & NODE_OPENCL_TYPE_NEED_IMGSIZE) {
				kernelArgIndex += 2;
			}
			if (data->isDelayed) {
				// needs to set opencl_buffer everytime when the buffer is part of a delay object
				if (agoGpuOclDataSetBufferAsKernelArg(data, opencl_kernel, kernelArgIndex, group) < 0)
					return -1;
			}
			else if (data->u.img.enableUserBufferOpenCL && data->opencl_buffer) {
				// need to set opencl_buffer and opencl_buffer_offset everytime if enableUserBufferOpenCL is true
				if (agoGpuOclDataSetBufferAsKernelArg(data, opencl_kernel, kernelArgIndex, group) < 0)
					return -1;
				err = clSetKernelArg(opencl_kernel, (cl_uint)kernelArgIndex + 2, sizeof(data->opencl_buffer_offset), &data->opencl_buffer_offset);
				if (err) {
					agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clSetKernelArg(supernode,%d,*,offset) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex, err, group);
					return -1;
				}
			}
			kernelArgIndex += 3;
			if (need_read_access) {
				auto dataToSync = data->u.img.isROI ? data->u.img.roiMasterImage : data;
				if (!(dataToSync->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
					if (dataToSync->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE | AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT)) {
						int64_t stime = agoGetClockCounter();
						if (dataToSync->opencl_buffer) {
							cl_int err = clEnqueueWriteBuffer(opencl_cmdq, dataToSync->opencl_buffer, CL_TRUE, dataToSync->opencl_buffer_offset, dataToSync->size, dataToSync->buffer, 0, NULL, NULL);
							if (err) { 
								agoAddLogEntry(&graph->ref, VX_FAILURE, "ERROR: clEnqueueWriteBuffer() => %d\n", err);
								return -1; 
							}
						}
						dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
						int64_t etime = agoGetClockCounter();
						graph->opencl_perf.buffer_write += etime - stime;
#if ENABLE_DEBUG_DUMP_CL_BUFFERS
						char fileName[128]; sprintf(fileName, "input_%%04d_%dx%d.yuv", dataToSync->u.img.width, dataToSync->u.img.height);
						clDumpBuffer(fileName, opencl_cmdq, dataToSync);
#endif
					}
				}
			}
		}
	}
	else if (data->ref.type == VX_TYPE_ARRAY) {
		if (data->isDelayed) {
			// needs to set opencl_buffer everytime when the buffer is part of a delay object
			if (agoGpuOclDataSetBufferAsKernelArg(data, opencl_kernel, kernelArgIndex, group) < 0)
				return -1;
		}
		kernelArgIndex += 3;
		if (need_read_access) {
			if (!(data->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
				if (data->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE | AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT)) {
					int64_t stime = agoGetClockCounter();
					vx_size size = data->u.arr.numitems * data->u.arr.itemsize;
					if (size > 0 && data->opencl_buffer) {
						cl_int err = clEnqueueWriteBuffer(opencl_cmdq, data->opencl_buffer, CL_TRUE, data->opencl_buffer_offset, size, data->buffer, 0, NULL, NULL);
						if (err) { 
							agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clEnqueueWriteBuffer() => %d (array)\n", err);
							return -1;
						}
					}
					data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
					int64_t etime = agoGetClockCounter();
					graph->opencl_perf.buffer_write += etime - stime;
#if ENABLE_DEBUG_DUMP_CL_BUFFERS
					clDumpBuffer("input_%04d.bin", opencl_cmdq, data);
#endif
				}
			}
		}
		if (need_read_access || !need_atomic_access) {
			// set numitems of the array
			err = clSetKernelArg(opencl_kernel, (cl_uint)kernelArgIndex - 1, sizeof(vx_uint32), &data->u.arr.numitems);
			if (err) { 
				agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clSetKernelArg(supernode,%d,*,numitems) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex - 1, err, group);
				return -1; 
			}
		}
	}
	else if (data->ref.type == AGO_TYPE_CANNY_STACK) {
		if (data->isDelayed) {
			// needs to set opencl_buffer everytime when the buffer is part of a delay object
			if (agoGpuOclDataSetBufferAsKernelArg(data, opencl_kernel, kernelArgIndex, group) < 0)
				return -1;
		}
		kernelArgIndex += 3;
		if (need_read_access) {
			agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: agoGpuOclDataSyncInputs: doesn't support object type %s for read-access in group#%d for kernel arg setting\n", agoEnum2Name(data->ref.type), group);
			return -1;
		}
	}
	else if (data->ref.type == VX_TYPE_THRESHOLD) {
		size_t size = sizeof(cl_uint);
		cl_uint2 value;
		value.s0 = data->u.thr.threshold_lower;
		if (data->u.thr.thresh_type == VX_THRESHOLD_TYPE_RANGE) {
			size = sizeof(cl_uint2);
			value.s1 = data->u.thr.threshold_upper;
		}
		err = clSetKernelArg(opencl_kernel, (cl_uint)kernelArgIndex, size, &value);
		if (err) { 
			agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clSetKernelArg(supernode,%d,%d,threshold) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex, (int)size, err, group);
			return -1; 
		}
		kernelArgIndex++;
	}
	else if (data->ref.type == VX_TYPE_SCALAR) {
		err = clSetKernelArg(opencl_kernel, (cl_uint)kernelArgIndex, sizeof(data->u.scalar.u.i), &data->u.scalar.u.i);
		if (err) { 
			agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clSetKernelArg(supernode,%d,*,scalar) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex, err, group);
			return -1; 
		}
		kernelArgIndex++;
	}
	else if (data->ref.type == VX_TYPE_MATRIX) {
		err = clSetKernelArg(opencl_kernel, (cl_uint)kernelArgIndex, data->size, data->buffer);
		if (err) { 
			agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clSetKernelArg(supernode,%d,*,matrix) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex, err, group);
			return -1; 
		}
		kernelArgIndex++;
	}
	else if (data->ref.type == VX_TYPE_CONVOLUTION) {
		err = clSetKernelArg(opencl_kernel, (cl_uint)kernelArgIndex, data->size << 1, data->reserved);
		if (err) { 
			agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clSetKernelArg(supernode,%d,*,convolution) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex, err, group);
			return -1; 
		}
		kernelArgIndex++;
	}
	else if (data->ref.type == VX_TYPE_LUT) {
		if (need_access) { // only use lut objects that need read access
			if (data->isDelayed) {
				// needs to set opencl_buffer everytime when the buffer is part of a delay object
				if (agoGpuOclDataSetBufferAsKernelArg(data, opencl_kernel, kernelArgIndex, group) < 0)
					return -1;
			}
			kernelArgIndex += 1;
			if (need_read_access) {
				if (!(data->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
					if (data->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE | AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT)) {
						int64_t stime = agoGetClockCounter();
						size_t origin[3] = { 0, 0, 0 };
						size_t region[3] = { 256, 1, 1 };
						err = clEnqueueWriteImage(opencl_cmdq, data->opencl_buffer, CL_TRUE, origin, region, 256, 0, data->buffer, 0, NULL, NULL);
						if (err) { 
							agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clEnqueueWriteImage(lut) => %d\n", err);
							return -1; 
						}
						data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
						int64_t etime = agoGetClockCounter();
						graph->opencl_perf.buffer_write += etime - stime;
					}
				}
			}
		}
	}
	else if (data->ref.type == VX_TYPE_REMAP) {
		if (need_access) { // only use image objects that need read access
			if (data->isDelayed) {
				// needs to set opencl_buffer everytime when the buffer is part of a delay object
				if (agoGpuOclDataSetBufferAsKernelArg(data, opencl_kernel, kernelArgIndex, group) < 0)
					return -1;
			}
			kernelArgIndex += 2;
			if (need_read_access) {
				if (!(data->buffer_sync_flags & AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED)) {
					if (data->buffer_sync_flags & (AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE | AGO_BUFFER_SYNC_FLAG_DIRTY_BY_COMMIT)) {
						int64_t stime = agoGetClockCounter();
						cl_int err = clEnqueueWriteBuffer(opencl_cmdq, data->opencl_buffer, CL_TRUE, data->opencl_buffer_offset, data->size, data->buffer, 0, NULL, NULL);
						if (err) { 
							agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clEnqueueWriteBuffer() => %d\n", err);
							return -1; 
						}
						data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_SYNCHED;
						int64_t etime = agoGetClockCounter();
						graph->opencl_perf.buffer_write += etime - stime;
					}
				}
			}
		}
	}
	else {
		agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: agoGpuOclDataSyncInputs: doesn't support object type %s in group#%d for kernel arg setting\n", agoEnum2Name(data->ref.type), group);
		return -1;
	}
	return 0;
}

static int agoGpuOclDataOutputMarkDirty(AgoGraph * graph, AgoData * data, bool need_access, bool need_write_access)
{
	if (data->ref.type == VX_TYPE_IMAGE) {
		if (need_access) { // only use image objects that need write access
			if (need_write_access) {
				auto dataToSync = data->u.img.isROI ? data->u.img.roiMasterImage : data;
				dataToSync->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
				dataToSync->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL;
			}
		}
	}
	else if (data->ref.type == VX_TYPE_ARRAY) {
		if (need_access) { // only use image objects that need write access
			if (need_write_access) {
				data->buffer_sync_flags &= ~AGO_BUFFER_SYNC_FLAG_DIRTY_MASK;
				data->buffer_sync_flags |= AGO_BUFFER_SYNC_FLAG_DIRTY_BY_NODE_CL;
			}
		}
	}
	return 0;
}

static int agoGpuOclDataOutputAtomicSync(AgoGraph * graph, AgoData * data)
{
	cl_command_queue opencl_cmdq = graph->opencl_cmdq ? graph->opencl_cmdq : graph->ref.context->opencl_cmdq;

	if (data->ref.type == VX_TYPE_ARRAY) {
#if ENABLE_DEBUG_DUMP_CL_BUFFERS
		clDumpBuffer("output_%04d_array.bin", opencl_cmdq, data);
		//printf("Press ENTER to continue... ");  char line[256]; gets(line);
#endif
		// update number of items
		cl_int err = CL_SUCCESS;
		int64_t stime = agoGetClockCounter();
		vx_uint32 * pNumItems = (vx_uint32 *)data->opencl_svm_buffer;
		if (data->opencl_buffer) {
			pNumItems = (vx_uint32 *)clEnqueueMapBuffer(opencl_cmdq, data->opencl_buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(vx_uint32), 0, NULL, NULL, &err);
			if (err) { 
				agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clEnqueueMapBuffer() for numitems => %d\n", err);
				return -1; 
			}
		}
		int64_t etime = agoGetClockCounter();
		graph->opencl_perf.buffer_read += etime - stime;
		// read and reset the counter
		data->u.arr.numitems = *pNumItems;
		*pNumItems = 0;
		if (data->opencl_buffer) {
			// unmap
			stime = agoGetClockCounter();
			err = clEnqueueUnmapMemObject(opencl_cmdq, data->opencl_buffer, pNumItems, 0, NULL, NULL);
			if (err) { 
				agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clEnqueueUnmapMemObject() for numitems => %d\n", err);
				return -1; 
			}
			etime = agoGetClockCounter();
			graph->opencl_perf.buffer_write += etime - stime;
		}
	}
	else if (data->ref.type == AGO_TYPE_CANNY_STACK) {
		// update number of items and reset it for next use
		int64_t stime = agoGetClockCounter();
		cl_int err = CL_SUCCESS;
		vx_uint8 * stack = data->opencl_svm_buffer;
		if (data->opencl_buffer) {
			stack = (vx_uint8 *)clEnqueueMapBuffer(opencl_cmdq, data->opencl_buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(vx_uint32), 0, NULL, NULL, &err);
			if (err) { 
				agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clEnqueueMapBuffer() for stacktop => %d\n", err);
				return -1; 
			}
		}
		int64_t etime = agoGetClockCounter();
		graph->opencl_perf.buffer_read += etime - stime;
		data->u.cannystack.stackTop = *(vx_uint32 *)stack;
		*(vx_uint32 *)stack = 0;
		if (data->opencl_buffer) {
			stime = agoGetClockCounter();
			err = clEnqueueUnmapMemObject(opencl_cmdq, data->opencl_buffer, stack, 0, NULL, NULL);
			if (err) { 
				agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clEnqueueUnmapMemObject() for stacktop => %d\n", err);
				return -1; 
			}
			etime = agoGetClockCounter();
			graph->opencl_perf.buffer_write += etime - stime;
			// read data
			if (data->u.cannystack.stackTop > 0) {
				int64_t stime = agoGetClockCounter();
				err = clEnqueueReadBuffer(opencl_cmdq, data->opencl_buffer, CL_TRUE, data->opencl_buffer_offset, data->u.cannystack.stackTop * sizeof(ago_coord2d_ushort_t), data->buffer, 0, NULL, NULL);
				if (err) { 
					agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: clEnqueueWriteBuffer() => %d (stacktop)\n", err);
					return -1; 
				}
				int64_t etime = agoGetClockCounter();
				graph->opencl_perf.buffer_read += etime - stime;
			}
		}
	}
	return 0;
}

static std::string agoGpuOclData2Decl(AgoData * data, vx_uint32 index, vx_uint32 dataFlags, vx_uint32 group)
{
	std::string code;
	char item[256];
	// add the object to argument
	if (data->ref.type == VX_TYPE_IMAGE) {
		if (dataFlags & NODE_OPENCL_TYPE_NEED_IMGSIZE) {
			sprintf(item, "uint p%d_width, uint p%d_height, ", index, index);
			code += item;
		}
		sprintf(item, "__global uchar * p%d_buf, uint p%d_stride, uint p%d_offset", index, index, index);
		code += item;
		if (dataFlags & DATA_OPENCL_FLAG_NEED_LOCAL) {
			sprintf(item, ", __local uchar * p%d_lbuf", index);
			code += item;
		}
	}
	else if (data->ref.type == VX_TYPE_ARRAY) {
		sprintf(item, "__global uchar * p%d_buf, uint p%d_offset, uint p%d_numitems", index, index, index);
		code += item;
	}
	else if (data->ref.type == VX_TYPE_SCALAR) {
		sprintf(item, "%s p%d", (data->u.scalar.type == VX_TYPE_FLOAT32) ? "float" : "uint", index);
		code += item;
	}
	else if (data->ref.type == VX_TYPE_THRESHOLD) {
		sprintf(item, "%s p%d", (data->u.thr.thresh_type == VX_THRESHOLD_TYPE_RANGE) ? "uint2" : "uint", index);
		code += item;
	}
	else if (data->ref.type == VX_TYPE_MATRIX) {
		if (data->u.mat.columns == 2 && data->u.mat.rows == 3) {
			sprintf(item, "ago_affine_matrix_t p%d", index);
			code += item;
		}
		else if (data->u.mat.columns == 3 && data->u.mat.rows == 3) {
			sprintf(item, "ago_perspective_matrix_t p%d", index);
			code += item;
		}
		else {
			agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: agoGpuOclData2Decl: doesn't support " VX_FMT_SIZE "x" VX_FMT_SIZE " matrix in group#%d for kernel declaration\n", data->u.mat.columns, data->u.mat.rows, group);
		}
	}
	else if (data->ref.type == VX_TYPE_CONVOLUTION) {
		sprintf(item, "COEF_" VX_FMT_SIZE "x" VX_FMT_SIZE " p%d", data->u.conv.columns, data->u.conv.rows, index);
		code += item;
	}
	else if (data->ref.type == VX_TYPE_LUT) {
		sprintf(item, "__read_only image1d_t p%d", index);
		code += item;
	}
	else if (data->ref.type == VX_TYPE_REMAP) {
		sprintf(item, "__global uchar * p%d_buf, uint p%d_stride", index, index);
		code += item;
	}
	else {
		agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: agoGpuOclData2Decl: doesn't support object type %s in group#%d for kernel declaration\n", agoEnum2Name(data->ref.type), group);
	}
	return code;
}

int agoGpuOclSuperNodeFinalize(AgoGraph * graph, AgoSuperNode * supernode)
{
	// make sure that all output images have same dimensions
	// check to make sure that max input hierarchy level is less than min output hierarchy level
	vx_uint32 width = 0, height = 0;
	vx_uint32 max_input_hierarchical_level = 0, min_output_hierarchical_level = (1 << 30);
	for (size_t index = 0; index < supernode->dataList.size(); index++) {
		AgoData * data = supernode->dataList[index];
		if (data->ref.type == VX_TYPE_IMAGE && supernode->dataInfo[index].argument_usage[VX_INPUT] == 0) {
			if (!width || !height) {
				width = data->u.img.width;
				height = data->u.img.height;
			}
			else if (width != data->u.img.width || height != data->u.img.height) {
				agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: agoGpuOclSuperNodeFinalize: doesn't support different image dimensions inside same group#%d\n", supernode->group);
				return -1;
			}
		}
		if (data->isVirtual && data->ref.type != VX_TYPE_SCALAR &&
			data->inputUsageCount == supernode->dataInfo[index].argument_usage[VX_INPUT] &&
			data->outputUsageCount == supernode->dataInfo[index].argument_usage[VX_OUTPUT] &&
			data->inoutUsageCount == supernode->dataInfo[index].argument_usage[VX_BIDIRECTIONAL])
		{
			// no need of this parameter as an argument into the kernel
			// mark that this will be an internal variable for the kernel
			supernode->dataInfo[index].needed_as_a_kernel_argument = false;
			// TBD: mark this the buffer doesn't need allocation
		}
		if (data->hierarchical_level > min_output_hierarchical_level) min_output_hierarchical_level = data->hierarchical_level;
		if (data->hierarchical_level < max_input_hierarchical_level) max_input_hierarchical_level = data->hierarchical_level;
	}
	if (max_input_hierarchical_level > min_output_hierarchical_level) {
		agoAddLogEntry(&graph->ref, VX_FAILURE, "ERROR: agoGpuOclSuperNodeFinalize: doesn't support mix of hierarchical levels inside same group#%d\n", supernode->group);
		return -1;
	}
	// decide work group dimensions (256 work-items)
	vx_uint32 work_group_width = AGO_OPENCL_WORKGROUP_SIZE_0;
	vx_uint32 work_group_height = AGO_OPENCL_WORKGROUP_SIZE_1;
	// save image size and compute global work
	//   - each work item processes 8x1 pixels
	supernode->width = width;
	supernode->height = height;
	supernode->opencl_global_work[0] = (((width + 7) >> 3) + (work_group_width  - 1)) & ~(work_group_width  - 1);
	supernode->opencl_global_work[1] = (  height           + (work_group_height - 1)) & ~(work_group_height - 1);
	for (size_t index = 0; index < supernode->dataList.size(); index++) {
		AgoData * data = supernode->dataList[index];
	}
	// clear the data flags
	for (size_t index = 0; index < supernode->dataList.size(); index++) {
		supernode->dataInfo[index].data_type_flags = 0;
	}
	for (size_t index = 0; index < supernode->dataList.size(); index++) {
		AgoData * data = supernode->dataList[index];
	}
	// generate code: node functions in OpenCL
	char item[256];
	std::string code = OPENCL_FORMAT(
		"#pragma OPENCL EXTENSION cl_amd_media_ops : enable\n"
		"#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable\n"
		"float4 amd_unpack(uint src)\n"
		"{\n"
		"  return (float4)(amd_unpack0(src), amd_unpack1(src), amd_unpack2(src), amd_unpack3(src));\n"
		"}\n"
		"\n"
		"///////////////////////////////////////////////////////////////////////////////\n"
		"// Data Types\n"
		"typedef uchar   U1x8;\n"
		"typedef uint2   U8x8;\n"
		"typedef  int4  S16x8;\n"
		"typedef uint4  U16x8;\n"
		"typedef uint8  U24x8;\n"
		"typedef uint8  U32x8;\n"
		"typedef float8 F32x8;\n"
		"typedef struct {\n"
		"  float M[3][2];\n"
		"} ago_affine_matrix_t;\n"
		"typedef struct {\n"
		"  float M[3][3];\n"
		"} ago_perspective_matrix_t;\n"
		"\n"
		"///////////////////////////////////////////////////////////////////////////////\n"
		"// load/store data\n"
		"void load_U1x8(U1x8 * r, uint x, uint y, __global uchar * p, uint stride)\n"
		"{\n"
		"  p += y*stride + (x >> 3);\n"
		"  *r = *((__global U1x8 *) p);\n"
		"}\n"
		"\n"
		"void load_U8x8(U8x8 * r, uint x, uint y, __global uchar * p, uint stride)\n"
		"{\n"
		"  p += y*stride + x;\n"
		"  *r = *((__global U8x8 *) p);\n"
		"}\n"
		"\n"
		"void load_S16x8(S16x8 * r, uint x, uint y, __global uchar * p, uint stride)\n"
		"{\n"
		"  p += y*stride + x + x;\n"
		"  *r = *((__global S16x8 *) p);\n"
		"}\n"
		"\n"
		"void load_U16x8(U16x8 * r, uint x, uint y, __global uchar * p, uint stride)\n"
		"{\n"
		"  p += y*stride + x + x;\n"
		"  *r = *((__global U16x8 *) p);\n"
		"}\n"
		"\n"
		"void load_U24x8(U24x8 * r, uint x, uint y, __global uchar * p, uint stride)\n"
		"{\n"
		"  p += y*stride + x * 3;\n"
		"  (*r).s012 = *((__global uint3 *)(p + 0));\n"
		"  (*r).s345 = *((__global uint3 *)(p + 12));\n"
		"}\n"
		"\n"
		"void load_U32x8(U32x8 * r, uint x, uint y, __global uchar * p, uint stride)\n"
		"{\n"
		"  p += y*stride + (x << 2);\n"
		"  *r = *((__global U32x8 *) p);\n"
		"}\n"
		"\n"
		"void load_F32x8(F32x8 * r, uint x, uint y, __global uchar * p, uint stride)\n"
		"{\n"
		"  p += y*stride + (x << 2);\n"
		"  *r = *((__global F32x8 *) p);\n"
		"}\n"
		"\n"
		"void store_U1x8(U1x8 r, uint x, uint y, __global uchar * p, uint stride)\n"
		"{\n"
		"  p += y*stride + (x >> 3);\n"
		"  *((__global U1x8 *)p) = r;\n"
		"}\n"
		"\n"
		"void store_U8x8(U8x8 r, uint x, uint y, __global uchar * p, uint stride)\n"
		"{\n"
		"  p += y*stride + x;\n"
		"  *((__global U8x8 *)p) = r;\n"
		"}\n"
		"\n"
		"void store_S16x8(S16x8 r, uint x, uint y, __global uchar * p, uint stride)\n"
		"{\n"
		"  p += y*stride + x + x;\n"
		"  *((__global S16x8 *)p) = r;\n"
		"}\n"
		"\n"
		"void store_U16x8(U16x8 r, uint x, uint y, __global uchar * p, uint stride)\n"
		"{\n"
		"  p += y*stride + x + x;\n"
		"  *((__global U16x8 *)p) = r;\n"
		"}\n"
		"\n"
		"void store_U24x8(U24x8 r, uint x, uint y, __global uchar * p, uint stride)\n"
		"{\n"
		"  p += y*stride + x * 3;\n"
		"  *((__global uint3 *)(p + 0)) = r.s012;\n"
		"  *((__global uint3 *)(p + 12)) = r.s345;\n"
		"}\n"
		"\n"
		"void store_U32x8(U32x8 r, uint x, uint y, __global uchar * p, uint stride)\n"
		"{\n"
		"  p += y*stride + (x << 2);\n"
		"  *((__global U32x8 *)p) = r;\n"
		"}\n"
		"\n"
		"void store_F32x8(F32x8 r, uint x, uint y, __global uchar * p, uint stride)\n"
		"{\n"
		"  p += y*stride + (x << 2);\n"
		"  *((__global F32x8 *)p) = r;\n"
		"}\n"
		"\n"
		"void Convert_U8_U1 (U8x8 * p0, U1x8 p1)\n"
		"{\n"
		"	U8x8 r;\n"
		"	r.s0  = (-(p1 &   1)) & 0x000000ff;\n"
		"	r.s0 |= (-(p1 &   2)) & 0x0000ff00;\n"
		"	r.s0 |= (-(p1 &   4)) & 0x00ff0000;\n"
		"	r.s0 |= (-(p1 &   8)) & 0xff000000;\n"
		"	r.s1  = (-((p1 >> 4) & 1)) & 0x000000ff;\n"
		"	r.s1 |= (-(p1 &  32)) & 0x0000ff00;\n"
		"	r.s1 |= (-(p1 &  64)) & 0x00ff0000;\n"
		"	r.s1 |= (-(p1 & 128)) & 0xff000000;\n"
		"	*p0 = r;\n"
		"}\n"
		"\n"
		"void Convert_U1_U8 (U1x8 * p0, U8x8 p1)\n"
		"{\n"
		"	U1x8 r;\n"
		"	r  =  p1.s0        &   1;\n"
		"	r |= (p1.s0 >>  7) &   2;\n"
		"	r |= (p1.s0 >> 14) &   4;\n"
		"	r |= (p1.s0 >> 21) &   8;\n"
		"	r |= (p1.s1 <<  4) &  16;\n"
		"	r |= (p1.s1 >>  3) &  32;\n"
		"	r |= (p1.s1 >> 10) &  64;\n"
		"	r |= (p1.s1 >> 17) & 128;\n"
		"	*p0 = r;\n"
		"}\n"
		);
	for (size_t index = 0; index < supernode->nodeList.size(); index++) {
		// get node and set node name
		AgoNode * node = supernode->nodeList[index];
		sprintf(node->opencl_name, "_n7%04d6f", (int)index ^ 3123);
		// generate kernel function code
		int status = VX_ERROR_NOT_IMPLEMENTED;
		if (node->akernel->func) {
			node->opencl_code = "";
			status = node->akernel->func(node, ago_kernel_cmd_opencl_codegen);
		}
		else if (node->akernel->opencl_codegen_callback_f) {
			// generation function declaration
			std::string code2;
			char item[256];
			sprintf(item, "void %s(", node->opencl_name); code2 = item;
			for (vx_uint32 i = 0; i < node->paramCount; i++) {
				AgoData * data = node->paramList[i];
				if (data) {
					if (i) code2 += ", ";
					size_t data_index = std::find(supernode->dataList.begin(), supernode->dataList.end(), data) - supernode->dataList.begin();
					if (data->ref.type == VX_TYPE_IMAGE) {
						if (node->akernel->argConfig[i] & AGO_KERNEL_ARG_INPUT_FLAG) {
							code2 += "uint x, uint y";
							sprintf(item, ", __global uchar * p%d_buf, uint p%d_stride", (int)data_index, (int)data_index);
							code2 += item;
							sprintf(item, ", uint p%d_width, uint p%d_height", (int)data_index, (int)data_index);
							code2 += item;
						}
						else {
							const char * reg_type = agoGpuImageFormat2RegType(data->u.img.format);
							sprintf(item, "%s p%d", reg_type, (int)data_index);
							code2 += item;
						}
					}
					else if (data->ref.type == VX_TYPE_REMAP) {
						sprintf(item, "__global uchar * p%d_buf, uint p%d_stride", (int)data_index, (int)data_index);
						code2 += item;
					}
					else {
						agoAddLogEntry(&data->ref, VX_FAILURE, "ERROR: agoGpuOclSuperNodeFinalize: doesn't support object type %s in group#%d for kernel declaration\n", agoEnum2Name(data->ref.type), supernode->group);
						return -1;
					}
				}
			}
			code2 += "\n";
			// generate function code
			node->opencl_code = code2;
			node->opencl_type = NODE_OPENCL_TYPE_MEM2REG | NODE_OPENCL_TYPE_NEED_IMGSIZE;
			node->opencl_param_mem2reg_mask = 0;
			node->opencl_param_discard_mask = 0;
			node->opencl_param_atomic_mask = 0;
			node->opencl_compute_work_multiplier = 0;
			node->opencl_compute_work_param_index = 0;
			node->opencl_output_array_param_index_plus1 = 0;
			node->opencl_local_buffer_usage_mask = 0;
			node->opencl_local_buffer_size_in_bytes = 0;
			vx_uint32 work_dim = 2;
			vx_size global_work[3] = { supernode->opencl_global_work[0], supernode->opencl_global_work[1], 1 };
			vx_size local_work[3] = { work_group_width, work_group_height, 1 };
			status = node->akernel->opencl_codegen_callback_f(node, true, node->opencl_name, node->opencl_code, node->opencl_build_options, work_dim, global_work, 
				local_work, node->opencl_local_buffer_usage_mask, node->opencl_local_buffer_size_in_bytes);
		}
		if (status != VX_SUCCESS) {
			agoAddLogEntry(&node->ref, VX_FAILURE, "ERROR: agoGpuOclSuperNodeFinalize: kernel %s in group#%d is not supported yet\n", node->akernel->name, supernode->group);
			return -1;
		}
		code += node->opencl_code;
		// update dataFlags[] if needed
		if (node->opencl_type & (NODE_OPENCL_TYPE_REG2REG | NODE_OPENCL_TYPE_MEM2REG)) {
			node->opencl_param_mem2reg_mask = 0;
			for (vx_uint32 i = 0; i < node->paramCount; i++) {
				AgoData * data = node->paramList[i];
				if (data) {
					if (node->opencl_param_discard_mask & (1 << i)) {
						// when code generator asked to discard this argument, mark that this argument is not needed anymore
						size_t data_index = std::find(supernode->dataList.begin(), supernode->dataList.end(), data) - supernode->dataList.begin();
						supernode->dataInfo[data_index].data_type_flags |= DATA_OPENCL_FLAG_DISCARD_PARAM;
					}
					else if (data->ref.type == VX_TYPE_IMAGE) {
						if (node->parameters[i].direction != VX_OUTPUT) {
							size_t data_index = std::find(supernode->dataList.begin(), supernode->dataList.end(), data) - supernode->dataList.begin();
							supernode->dataInfo[data_index].data_type_flags |= (node->opencl_type & (NODE_OPENCL_TYPE_REG2REG | NODE_OPENCL_TYPE_MEM2REG | NODE_OPENCL_TYPE_NEED_IMGSIZE));
							if (i > 0) {
								if ((node->opencl_local_buffer_size_in_bytes > 0) && (node->opencl_local_buffer_usage_mask & (1 << i))) {
									// mark that local data buffer is needed and specify the buffer size
									supernode->dataInfo[data_index].data_type_flags |= DATA_OPENCL_FLAG_NEED_LOCAL;
									supernode->dataInfo[data_index].local_buffer_size_in_bytes = node->opencl_local_buffer_size_in_bytes;
								}
								if (node->opencl_type & NODE_OPENCL_TYPE_MEM2REG) {
									// mark that the image has NODE_OPENCL_TYPE_MEM2REG
									node->opencl_param_mem2reg_mask = (1 << i);
								}
							}
						}
					}
				}
			}
		}
	}
	// generate code: kernel declaration
	sprintf(item, OPENCL_FORMAT("__kernel __attribute__((reqd_work_group_size(%d, %d, 1)))\nvoid %s(uint width, uint height"), work_group_width, work_group_height, NODE_OPENCL_KERNEL_NAME);
	code += item;
#if ENABLE_LOCAL_DEBUG_MESSAGES
	printf("===> *** supernode-%d has dataList.size()=%d\n", supernode->group, supernode->dataList.size());
#endif
	for (size_t index = 0, line_length = 0; index < supernode->dataList.size(); index++) {
		AgoData * data = supernode->dataList[index];
#if ENABLE_LOCAL_DEBUG_MESSAGES
		printf("===> karg[%d] = { %d, 0x%08x, [ %2d %2d %2d ], %5d } -- %s\n", index, supernode->dataInfo[index].needed_as_a_kernel_argument, supernode->dataInfo[index].data_type_flags, supernode->dataInfo[index].argument_usage[0], supernode->dataInfo[index].argument_usage[1], supernode->dataInfo[index].argument_usage[2], supernode->dataInfo[index].local_buffer_size_in_bytes, data->name.c_str());
#endif
		if (supernode->dataInfo[index].needed_as_a_kernel_argument && !(supernode->dataInfo[index].data_type_flags & DATA_OPENCL_FLAG_DISCARD_PARAM)) { // only use objects that need read/write access
			// add the object to argument
			std::string arg = agoGpuOclData2Decl(data, (vx_uint32)index, supernode->dataInfo[index].data_type_flags & ~DATA_OPENCL_FLAG_NEED_LOCAL, supernode->group);
			if (arg.length() > 0) {
				line_length += arg.length();
				if (line_length > 800) {
					// make sure that lines never exceed 1000 characters: assumption made by the CObfuscator
					code += "\n    ";
					line_length = 0;
				}
				code += ", ";
				code += arg;
				if (data->ref.type == VX_TYPE_IMAGE) {
					supernode->dataInfo[index].data_type_flags |= DATA_OPENCL_FLAG_BUFFER;
				}
			}
			else {
				return -1;
			}
		}
	}
	code += ")\n";
	// generate code: workitem (x,y) computation
	code += "{\n\tuint x = get_global_id(0) * 8;\n\tuint y = get_global_id(1);\n\tbool valid = (x < width) && (y < height);\n\n";
	// generate code: add offset to image address
	bool uses_local_memory = false;
	for (size_t index = 0; index < supernode->dataList.size(); index++) {
		AgoData * data = supernode->dataList[index];
		if (data->ref.type == VX_TYPE_IMAGE) {
			if (supernode->dataInfo[index].data_type_flags & DATA_OPENCL_FLAG_NEED_LOCAL) {
				sprintf(item, "\t__local uchar p%d_lbuf[%d];\n", (int)index, supernode->dataInfo[index].local_buffer_size_in_bytes);
				code += item;
				uses_local_memory = true;
			}
			if (supernode->dataInfo[index].data_type_flags & DATA_OPENCL_FLAG_BUFFER) {
				sprintf(item, "\tp%d_buf += p%d_offset;\n", (int)index, (int)index);
				code += item;
			}
			if (supernode->dataInfo[index].needed_as_a_kernel_argument) { // only use objects that need read/write access
				if (supernode->dataInfo[index].argument_usage[VX_INPUT] || supernode->dataInfo[index].argument_usage[VX_BIDIRECTIONAL]) {
					// mark that load is needed
					supernode->dataInfo[index].data_type_flags |= (DATA_OPENCL_FLAG_NEED_LOAD_R2R | DATA_OPENCL_FLAG_NEED_LOAD_M2R);
				}
			}
		}
	}
	if (!uses_local_memory) {
		code += "\tif (valid) {\n";
	}
	// generate code: declara register variables for images
	for (size_t index = 0; index < supernode->dataList.size(); index++) {
		AgoData * data = supernode->dataList[index];
		if (data->ref.type == VX_TYPE_IMAGE) {
			const char * reg_type = agoGpuImageFormat2RegType(data->u.img.format);
			sprintf(item, "\t\t%sx8 p%d;\n", reg_type, (int)index);
			code += item;
			if (supernode->dataInfo[index].needed_as_a_kernel_argument) { // only use objects that need read/write access
				if (supernode->dataInfo[index].argument_usage[VX_OUTPUT]) {
					// mark that load is not needed
					supernode->dataInfo[index].data_type_flags &= ~DATA_OPENCL_FLAG_NEED_LOAD_R2R;
				}
			}
		}
	}
	// generate code: actual computation
	for (size_t index = 0; index < supernode->nodeList.size(); index++) {
		AgoNode * node = supernode->nodeList[index];
		// issues all required loads
		for (vx_uint32 i = 0; i < node->paramCount; i++) {
			AgoData * data = node->paramList[i];
			if (data) {
				size_t data_index = std::find(supernode->dataList.begin(), supernode->dataList.end(), data) - supernode->dataList.begin();
				if ((supernode->dataInfo[data_index].data_type_flags & NODE_OPENCL_TYPE_REG2REG) && (supernode->dataInfo[data_index].data_type_flags & DATA_OPENCL_FLAG_NEED_LOAD_R2R)) {
					const char * reg_type = agoGpuImageFormat2RegType(data->u.img.format);
					sprintf(item, "\t\tload_%sx8(&p%d, x, y, p%d_buf, p%d_stride);\n", reg_type, (int)data_index, (int)data_index, (int)data_index);
					code += item;
					// mark that load has been issued
					supernode->dataInfo[data_index].data_type_flags &= ~DATA_OPENCL_FLAG_NEED_LOAD_R2R;
				}
			}
		}
		// generate computation
		sprintf(item, "\t\t%s(", node->opencl_name); code += item;
		for (vx_uint32 i = 0; i < node->paramCount; i++) {
			AgoData * data = node->paramList[i];
			if (data) {
				size_t data_index = std::find(supernode->dataList.begin(), supernode->dataList.end(), data) - supernode->dataList.begin();
				if (!(supernode->dataInfo[data_index].data_type_flags & DATA_OPENCL_FLAG_DISCARD_PARAM)) {
					if ((supernode->dataInfo[data_index].data_type_flags & NODE_OPENCL_TYPE_MEM2REG) && 
						(supernode->dataInfo[data_index].data_type_flags & DATA_OPENCL_FLAG_NEED_LOAD_M2R) &&
						(node->opencl_param_mem2reg_mask & (1 << i)))
					{
						code += ", x, y";
						if (node->opencl_local_buffer_usage_mask & (1 << i)) {
							sprintf(item, ", p%d_lbuf", (int)data_index);
							code += item;
						}
						sprintf(item, ", p%d_buf, p%d_stride", (int)data_index, (int)data_index);
						code += item;
						if (supernode->dataInfo[data_index].data_type_flags & NODE_OPENCL_TYPE_NEED_IMGSIZE) {
							sprintf(item, ", p%d_width, p%d_height", (int)data_index, (int)data_index);
							code += item;
						}
						// mark that load has been issued
						supernode->dataInfo[data_index].data_type_flags &= ~DATA_OPENCL_FLAG_NEED_LOAD_M2R;
					}
					else if (data->ref.type == VX_TYPE_REMAP) {
						sprintf(item, ", p%d_buf, p%d_stride", (int)data_index, (int)data_index);
						code += item;
					}
					else {
						sprintf(item, "%s%sp%d", i ? ", " : "", (node->akernel->argConfig[i] & AGO_KERNEL_ARG_OUTPUT_FLAG) ? "&" : "", (int)data_index);
						code += item;
					}
				}
			}
		}
		// end of function call with actual kernel name as a comment for debug
		code += "); // ";
		code += agoGpuGetKernelFunctionName(node);
		code += "\n";
	}
	if (uses_local_memory) {
		code += "\tif (valid) {\n";
	}
	// generate code: issue stores
	for (size_t index = 0; index < supernode->dataList.size(); index++) {
		AgoData * data = supernode->dataList[index];
		if (data->ref.type == VX_TYPE_IMAGE) {
			if (supernode->dataInfo[index].needed_as_a_kernel_argument &&
				(supernode->dataInfo[index].argument_usage[VX_OUTPUT] || supernode->dataInfo[index].argument_usage[VX_BIDIRECTIONAL]))
			{ // only use objects that need write access
				const char * reg_type = agoGpuImageFormat2RegType(data->u.img.format);
				sprintf(item, "\t\tstore_%sx8(p%d, x, y, p%d_buf, p%d_stride);\n", reg_type, (int)index, (int)index, (int)index);
				code += item;
			}
		}
	}
	// generate code: end of function and save
	code += "\t}\n}\n";
	supernode->opencl_code = code;
	const char * opencl_code = supernode->opencl_code.c_str();

	// dump OpenCL kernel if environment variable AGO_DUMP_GPU is specified with dump file path prefix
	// the output file name will be "$(AGO_DUMP_GPU)-<group>.cl"
	char textBuffer[1024];
	if (agoGetEnvironmentVariable("AGO_DUMP_GPU", textBuffer, sizeof(textBuffer))) {
		char fileName[1024];
		sprintf(fileName, "%s-%d.cl", textBuffer, supernode->group);
		FILE * fp = fopen(fileName, "w");
		if (!fp) agoAddLogEntry(NULL, VX_FAILURE, "ERROR: unable to create: %s\n", fileName);
		else {
			fprintf(fp, "%s", opencl_code);
			fclose(fp);
			agoAddLogEntry(NULL, VX_SUCCESS, "OK: created %s\n", fileName);
		}
	}

	// create compile the OpenCL code into OpenCL kernel object
	supernode->opencl_cmdq = graph->opencl_cmdq;
	cl_int err;
	supernode->opencl_program = clCreateProgramWithSource(graph->ref.context->opencl_context, 1, &opencl_code, NULL, &err);
	if (err) { 
		agoAddLogEntry(&graph->ref, VX_FAILURE, "ERROR: clCreateProgramWithSource(%p,1,*,NULL,*) failed(%d) for group#%d\n", graph->ref.context->opencl_context, err, supernode->group);
		return -1; 
	}
	std::string opencl_build_options = graph->ref.context->opencl_build_options;
	err = clBuildProgram(supernode->opencl_program, 1, &graph->opencl_device, opencl_build_options.c_str(), NULL, NULL);
	if (err) { 
		agoAddLogEntry(&graph->ref, VX_FAILURE, "ERROR: clBuildProgram(%p,%s) failed(%d) for group#%d\n", supernode->opencl_program, graph->ref.context->opencl_build_options, err, supernode->group);
#if _DEBUG // dump warnings/errors to console in debug build mode
		size_t logSize = 1024 * 1024; char * log = new char[logSize]; memset(log, 0, logSize);
		clGetProgramBuildInfo(supernode->opencl_program, graph->opencl_device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
		printf("<<<<\n%s\n>>>>\n", log);
		delete[] log;
#endif
		return -1;
	}
	supernode->opencl_kernel = clCreateKernel(supernode->opencl_program, NODE_OPENCL_KERNEL_NAME, &err);
	if (err) { 
		agoAddLogEntry(&graph->ref, VX_FAILURE, "ERROR: clCreateKernel(%p,supernode) failed(%d) for group#%d\n", supernode->opencl_program, err, supernode->group);
		return -1; 
	}
	// set all kernel objects
	vx_uint32 kernelArgIndex = 0;
	err = clSetKernelArg(supernode->opencl_kernel, (cl_uint)kernelArgIndex, sizeof(cl_uint), &width);
	if (err) { 
		agoAddLogEntry(&graph->ref, VX_FAILURE, "ERROR: clSetKernelArg(supernode,%d,*,width) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex, err, supernode->group);
		return -1; 
	}
	kernelArgIndex++;
	err = clSetKernelArg(supernode->opencl_kernel, (cl_uint)kernelArgIndex, sizeof(cl_uint), &height);
	if (err) { 
		agoAddLogEntry(&graph->ref, VX_FAILURE, "ERROR: clSetKernelArg(supernode,%d,*,height) failed(%d) for group#%d\n", (cl_uint)kernelArgIndex, err, supernode->group);
		return -1; 
	}
	kernelArgIndex++;
	for (size_t index = 0; index < supernode->dataList.size(); index++) {
		if (!(supernode->dataInfo[index].data_type_flags & DATA_OPENCL_FLAG_DISCARD_PARAM)) {
			bool need_access = supernode->dataInfo[index].needed_as_a_kernel_argument;
			if (agoGpuOclSetKernelArgs(supernode->opencl_kernel, kernelArgIndex, supernode->dataList[index], need_access, supernode->dataInfo[index].data_type_flags, supernode->group) < 0) {
				return -1;
			}
		}
	}
	return 0;
}

int agoGpuOclSuperNodeLaunch(AgoGraph * graph, AgoSuperNode * supernode)
{
	// make sure that all input buffers are synched and other arguments are updated
	vx_uint32 kernelArgIndex = 2;
	for (size_t index = 0; index < supernode->dataList.size(); index++) {
		if (!(supernode->dataInfo[index].data_type_flags & DATA_OPENCL_FLAG_DISCARD_PARAM)) {
			bool need_access = supernode->dataInfo[index].needed_as_a_kernel_argument;
			bool need_read_access = supernode->dataInfo[index].argument_usage[VX_INPUT] || supernode->dataInfo[index].argument_usage[VX_BIDIRECTIONAL];
			if (agoGpuOclDataInputSync(graph, supernode->opencl_kernel, kernelArgIndex, supernode->dataList[index], supernode->dataInfo[index].data_type_flags, supernode->group, need_access, need_read_access, false) < 0) {
				return -1;
			}
		}
	}
	// launch the kernel
	int64_t stime = agoGetClockCounter();
	cl_int err;
	err = clEnqueueNDRangeKernel(supernode->opencl_cmdq, supernode->opencl_kernel, 2, NULL, supernode->opencl_global_work, NULL, 0, NULL, &supernode->opencl_event);
	if (err) { 
		agoAddLogEntry(&graph->ref, VX_FAILURE, "ERROR: clEnqueueNDRangeKernel(supernode,2,*,%dx%d,...) failed(%d) for group#%d\n", (cl_uint)supernode->opencl_global_work[0], (cl_uint)supernode->opencl_global_work[1], err, supernode->group);
		return -1; 
	}
	err = clFlush(supernode->opencl_cmdq);
	if (err) { 
		agoAddLogEntry(&graph->ref, VX_FAILURE, "ERROR: clFlush(supernode) failed(%d) for group#%d\n", err, supernode->group);
		return -1; 
	}
	int64_t etime = agoGetClockCounter();
	graph->opencl_perf.kernel_enqueue += etime - stime;
	// mark that supernode outputs are dirty
	for (size_t index = 0; index < supernode->dataList.size(); index++) {
		if (!(supernode->dataInfo[index].data_type_flags & DATA_OPENCL_FLAG_DISCARD_PARAM)) {
			bool need_access = supernode->dataInfo[index].needed_as_a_kernel_argument;
			bool need_write_access = supernode->dataInfo[index].argument_usage[VX_OUTPUT] || supernode->dataInfo[index].argument_usage[VX_BIDIRECTIONAL];
			if (agoGpuOclDataOutputMarkDirty(graph, supernode->dataList[index], need_access, need_write_access) < 0) {
				return -1;
			}
		}
	}
	return 0;
}

int agoGpuOclSuperNodeWait(AgoGraph * graph, AgoSuperNode * supernode)
{
	// wait for completion
	int64_t stime = agoGetClockCounter();
	cl_int err;
	err = clWaitForEvents(1, &supernode->opencl_event);
	if (err) { 
		agoAddLogEntry(&graph->ref, VX_FAILURE, "ERROR: clWaitForEvents(1,%p) failed(%d) for group#%d\n", supernode->opencl_event, err, supernode->group);
		return -1; 
	}
	clReleaseEvent(supernode->opencl_event);
	supernode->opencl_event = NULL;
	int64_t etime = agoGetClockCounter();
	graph->opencl_perf.kernel_wait += etime - stime;
#if ENABLE_DEBUG_DUMP_CL_BUFFERS
	// dump supernode outputs
	for (size_t index = 0; index < supernode->dataList.size(); index++) {
		if (!(supernode->dataInfo[index].data_type_flags & DATA_OPENCL_FLAG_DISCARD_PARAM)) {
			bool need_access = supernode->dataInfo[index].needed_as_a_kernel_argument;
			bool need_write_access = supernode->dataInfo[index].argument_usage[VX_OUTPUT] || supernode->dataInfo[index].argument_usage[VX_BIDIRECTIONAL];
			auto data = supernode->dataList[index];
			if (data->ref.type == VX_TYPE_IMAGE) {
				if (need_access) { // only use image objects that need write access
					if (need_write_access) {
						auto dataToSync = data->u.img.isROI ? data->u.img.roiMasterImage : data;
						char fileName[128]; sprintf(fileName, "output_%%04d_%dx%d.yuv", dataToSync->u.img.width, dataToSync->u.img.height);
						cl_command_queue opencl_cmdq = graph->opencl_cmdq ? graph->opencl_cmdq : graph->ref.context->opencl_cmdq;
						clDumpBuffer(fileName, opencl_cmdq, dataToSync);
						//printf("Press ENTER to continue... ");  char line[256]; gets(line);
					}
				}
			}
		}
	}
#endif
	return 0;
}

int agoGpuOclSingleNodeFinalize(AgoGraph * graph, AgoNode * node)
{
	const char * opencl_code = node->opencl_code.c_str();

	// dump OpenCL kernel if environment variable AGO_DUMP_GPU is specified with dump file path prefix
	// the output file name will be "$(AGO_DUMP_GPU)-0.<counter>.cl"
	char textBuffer[1024];
	if (agoGetEnvironmentVariable("AGO_DUMP_GPU", textBuffer, sizeof(textBuffer))) {
		char fileName[1024]; static int counter = 0;
		sprintf(fileName, "%s-0.%04d.cl", textBuffer, counter++);
		FILE * fp = fopen(fileName, "w");
		if (!fp) agoAddLogEntry(NULL, VX_FAILURE, "ERROR: unable to create: %s\n", fileName);
		else {
			fprintf(fp, "%s", opencl_code);
			fclose(fp);
			agoAddLogEntry(NULL, VX_SUCCESS, "OK: created %s\n", fileName);
		}
	}

	// create compile the OpenCL code into OpenCL kernel object
	vx_context context = graph->ref.context;
	cl_int err;
	node->opencl_program = clCreateProgramWithSource(context->opencl_context, 1, &opencl_code, NULL, &err);
	if (err) { 
		agoAddLogEntry(&node->ref, VX_FAILURE, "ERROR: clCreateProgramWithSource(%p,1,*,NULL,*) failed(%d) for %s\n", context->opencl_context, err, node->akernel->name);
		return -1; 
	}
	err = clBuildProgram(node->opencl_program, 1, &graph->opencl_device, node->opencl_build_options.c_str(), NULL, NULL);
	if (err) {
		agoAddLogEntry(&node->ref, VX_FAILURE, "ERROR: clBuildProgram(%p,%s) failed(%d) for %s\n", node->opencl_program, node->opencl_build_options.c_str(), err, node->akernel->name);
#if _DEBUG // dump warnings/errors to console in debug build mode
		size_t logSize = 1024 * 1024; char * log = new char[logSize]; memset(log, 0, logSize);
		clGetProgramBuildInfo(node->opencl_program, graph->opencl_device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
		printf("<<<<\n%s\n>>>>\n", log);
		delete[] log;
#endif
		return -1;
	}
	node->opencl_kernel = clCreateKernel(node->opencl_program, node->opencl_name, &err);
	if (err) { 
		agoAddLogEntry(&node->ref, VX_FAILURE, "ERROR: clCreateKernel(%p,supernode) failed(%d) for %s\n", node->opencl_program, err, node->akernel->name);
		return -1; 
	}
	// set all kernel objects
	vx_uint32 kernelArgIndex = 0;
	for (size_t index = 0; index < node->paramCount; index++) {
		if (node->paramList[index] && !(node->opencl_param_discard_mask & (1 << index))) {
			vx_uint32 dataFlags = 0;
			if (node->paramList[index]->ref.type == VX_TYPE_IMAGE) {
				dataFlags |= NODE_OPENCL_TYPE_NEED_IMGSIZE;
			}
			else if (node->paramList[index]->ref.type == VX_TYPE_ARRAY) {
				if (node->opencl_param_atomic_mask & (1 << index)) {
					dataFlags |= NODE_OPENCL_TYPE_ATOMIC;
				}
			}
			if (agoGpuOclSetKernelArgs(node->opencl_kernel, kernelArgIndex, node->paramList[index], true, dataFlags, 0) < 0) {
				return -1;
			}
		}
	}
	return 0;
}

int agoGpuOclSingleNodeLaunch(AgoGraph * graph, AgoNode * node)
{
	// compute global work (if requested) and set numitems of output array (if requested further)
	if (node->opencl_compute_work_multiplier > 0) {
		AgoData * data = node->paramList[node->opencl_compute_work_param_index];
		if (data->ref.type == VX_TYPE_ARRAY) {
			// derive global_work[0] from numitems of array
			node->opencl_global_work[0] = data->u.arr.numitems * node->opencl_compute_work_multiplier;
			if (node->opencl_local_work[0] > 0) {
				size_t mask = node->opencl_local_work[0] - 1;
				node->opencl_global_work[0] = (node->opencl_global_work[0] + mask) & ~mask;
			}
			// set numitems of output array param index (if requested)
			if (node->opencl_output_array_param_index_plus1 > 0) {
				AgoData * arr = node->paramList[node->opencl_output_array_param_index_plus1 - 1];
				if (arr->ref.type == VX_TYPE_ARRAY) {
					arr->u.arr.numitems = data->u.arr.numitems;
				}
			}
		}
		else {
			agoAddLogEntry(&node->ref, VX_FAILURE, "ERROR: agoGpuOclSingleNodeLaunch: invalid opencl_compute_work_multiplier=%d\n", node->opencl_compute_work_multiplier);
			return -1;
		}
	}
	// make sure that all input buffers are synched and other arguments are updated
	vx_uint32 kernelArgIndex = 0;
	for (size_t index = 0; index < node->paramCount; index++) {
		if (node->paramList[index] && !(node->opencl_param_discard_mask & (1 << index))) {
			bool need_read_access = node->parameters[index].direction != VX_OUTPUT ? true : false;
			bool need_atomic_access = (node->opencl_param_atomic_mask & (1 << index)) ? true : false;
			if (agoGpuOclDataInputSync(graph, node->opencl_kernel, kernelArgIndex, node->paramList[index], NODE_OPENCL_TYPE_NEED_IMGSIZE, 0, true, need_read_access, need_atomic_access) < 0) {
				return -1;
			}
		}
	}
	// launch the kernel
	int64_t stime = agoGetClockCounter();
	cl_int err;
	err = clEnqueueNDRangeKernel(graph->opencl_cmdq, node->opencl_kernel, node->opencl_work_dim, NULL, node->opencl_global_work, NULL, 0, NULL, &node->opencl_event);
	if (err) { 
		agoAddLogEntry(&node->ref, VX_FAILURE, "ERROR: clEnqueueNDRangeKernel(supernode,%d,*,{%d,%d,%d},...) failed(%d) for %s\n", (cl_uint)node->opencl_work_dim, (cl_uint)node->opencl_global_work[0], (cl_uint)node->opencl_global_work[1], (cl_uint)node->opencl_global_work[2], err, node->akernel->name);
		return -1; 
	}
	err = clFlush(graph->opencl_cmdq);
	if (err) { 
		agoAddLogEntry(&node->ref, VX_FAILURE, "ERROR: clFlush(supernode) failed(%d) for %s\n", err, node->akernel->name);
		return -1; 
	}
	int64_t etime = agoGetClockCounter();
	graph->opencl_perf.kernel_enqueue += etime - stime;
	// mark that node outputs are dirty
	for (size_t index = 0; index < node->paramCount; index++) {
		if (node->paramList[index]) {
			bool need_write_access = node->parameters[index].direction != VX_INPUT ? true : false;
			if (agoGpuOclDataOutputMarkDirty(graph, node->paramList[index], true, need_write_access) < 0) {
				return -1;
			}
		}
	}
	return 0;
}

int agoGpuOclSingleNodeWait(AgoGraph * graph, AgoNode * node)
{
	// wait for completion
	int64_t stime = agoGetClockCounter();
	cl_int err;
	err = clWaitForEvents(1, &node->opencl_event);
	if (err) { 
		agoAddLogEntry(&node->ref, VX_FAILURE, "ERROR: clWaitForEvents(1,%p) failed(%d) for %s\n", node->opencl_event, err, node->akernel->name);
		return -1; 
	}
	clReleaseEvent(node->opencl_event);
	node->opencl_event = NULL;
	int64_t etime = agoGetClockCounter();
	graph->opencl_perf.kernel_wait += etime - stime;
	// sync the outputs
	for (size_t index = 0; index < node->paramCount; index++) {
		if (node->paramList[index]) {
			bool need_write_access = node->parameters[index].direction != VX_INPUT ? true : false;
			if (need_write_access && node->opencl_param_atomic_mask & (1 << index)) {
				if (agoGpuOclDataOutputAtomicSync(graph, node->paramList[index]) < 0) {
					return -1;
				}
			}
#if ENABLE_DEBUG_DUMP_CL_BUFFERS
			else if (node->paramList[index]->ref.type == VX_TYPE_IMAGE) {
				if (need_write_access) {
					auto dataToSync = node->paramList[index]->u.img.isROI ? node->paramList[index]->u.img.roiMasterImage : node->paramList[index];
					char fileName[128]; sprintf(fileName, "input_%%04d_%dx%d.yuv", dataToSync->u.img.width, dataToSync->u.img.height);
					cl_command_queue opencl_cmdq = graph->opencl_cmdq ? graph->opencl_cmdq : graph->ref.context->opencl_cmdq;
					clDumpBuffer(fileName, opencl_cmdq, node->paramList[index]);
					//printf("Press ENTER to continue... ");  char line[256]; gets(line);
				}
			}
#endif
		}
	}
	if (node->opencl_scalar_array_output_sync.enable && 
		node->paramList[node->opencl_scalar_array_output_sync.paramIndexScalar] && 
		node->paramList[node->opencl_scalar_array_output_sync.paramIndexArray])
	{
		// updated scalar with numitems of array
		node->paramList[node->opencl_scalar_array_output_sync.paramIndexScalar]->u.scalar.u.s =
			node->paramList[node->opencl_scalar_array_output_sync.paramIndexArray]->u.arr.numitems;
	}

	// The num items in an array should not exceed the capacity unless kernels need it for reporting number of items detected (ex. FAST corners)
	for (size_t index = 0; index < node->paramCount; index++) {
		if (node->paramList[index]) {
			bool need_write_access = node->parameters[index].direction != VX_INPUT ? true : false;
			if (need_write_access && node->opencl_param_atomic_mask & (1 << index)) {
				if (node->paramList[index]->ref.type == VX_TYPE_ARRAY) {
					node->paramList[index]->u.arr.numitems = min(node->paramList[index]->u.arr.numitems, node->paramList[index]->u.arr.capacity);
				}
			}
		}
	}
	return 0;
}

#endif

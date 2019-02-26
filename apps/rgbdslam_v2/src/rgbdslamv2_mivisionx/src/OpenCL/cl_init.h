#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <vector>
#include <opencv2/features2d/features2d.hpp>

#include "../parameter_server.h"


extern cl_platform_id platform_id;
extern cl_uint ret_num_platforms;
extern cl_device_id device_id;
extern cl_context context;
extern cl_program program;
extern cl_int binary_status;
extern cl_uint ret_num_devices;
extern size_t binary_size;
extern char *binary;
extern long length;
extern cl_int ret;

char* common_read_file(const char *path, long *length_out);
void init_opencl();
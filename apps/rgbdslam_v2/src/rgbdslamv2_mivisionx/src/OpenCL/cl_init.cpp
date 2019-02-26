#include "cl_init.h"
#include <stdio.h>

#define 	MEM_SIZE                (1500)
#define 	MAX_SOURCE_SIZE 	    (0xBB8000)

cl_platform_id platform_id;
cl_uint ret_num_platforms;
cl_device_id device_id;
cl_context context;
cl_program program;
cl_int binary_status;
cl_uint ret_num_devices;
size_t binary_size;
char *binary;
long length;
cl_int ret;

// Function to read binary file - taken from the following GitHub link
// https://github.com/cirosantilli/cpp-cheat/blob/b1e9696cb18a12c4a41e0287695a2a6591b04597/opencl/common.h
char* common_read_file(const char *path, long *length_out) 
{
    char *buffer;
    FILE *f;
    long length;

    f = fopen(path, "r");
    fseek(f, 0, SEEK_END);
    length = ftell(f);
    fseek(f, 0, SEEK_SET);
    buffer = (char *)malloc(length);
    if (fread(buffer, 1, length, f) < (size_t)length) 
    {
        return NULL;
    }
    fclose(f);
    if (NULL != length_out) 
    {
        *length_out = length;
    }
    return buffer;
}

void init_opencl()
{	
	platform_id = NULL;
	device_id = NULL;
	context = NULL;
	program = NULL;

	// Get directory of OpenCL kernel files from parameter server, which in turn gets it from the ROS launch file
	ParameterServer* ps = ParameterServer::instance();
	std::string kernel_path = ps->get<std::string>("opencl_kernel_path");

	// To make program from source, set use_source flag = true
	// To make program from binary, set use_binary flag = true
	// To make the binary, set use_source = true and make_binary = true
	// Change file paths accordingly
	bool make_binary = false;
	bool use_binary  = false;
	bool use_source  = true;

	// Get Platform ID, Device ID and create a common context which is to be used every time
	// an OpenCL kernel is executed

	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	if (ret != CL_SUCCESS) 
		printf("Failed to get platform ID.\n");

	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
	if (ret != CL_SUCCESS)
		printf("Failed to get device ID.\n");

	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, NULL);//&ret);
	if (ret != CL_SUCCESS)
		printf("Failed to create OpenCL context.\n");


	if (use_binary == true)
	{
		// Append file name to path
		std::string binary_path;
		binary_path.append(kernel_path);
		binary_path.append("bruteforce_cl.bin");
		binary = common_read_file(binary_path.c_str(), &length);
		binary_size = length;
		program = clCreateProgramWithBinary(context, 1, &device_id, &binary_size, (const unsigned char **)&binary, &binary_status, &ret);
	}

	if (use_source == true)
	{
		// Load the source code containing the kernel
		char string[MEM_SIZE];
		FILE *fp;
		char *source_str;
		size_t source_size; 

		// Append file name to path
		std::string fileName;
		fileName.append(kernel_path);
		fileName.append("bf.cl");

		fp = fopen(fileName.c_str(), "r");
		if (!fp) {
			fprintf(stderr, "Failed to load kernel.\n");
			exit(1);
		}
		source_str = (char*) malloc (MAX_SOURCE_SIZE);
		source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
		fclose(fp);

		// Create Kernel Program from source
		program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);
		if (ret != CL_SUCCESS)
			printf("Failed to create OpenCL program from source %d\n", (int) ret);
	}
	

	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if (ret != CL_SUCCESS) 
	{
		printf("Failed to build program %d\n", (int) ret);
		char build_log[16348];
		clGetProgramBuildInfo (program, device_id, CL_PROGRAM_BUILD_LOG, sizeof (build_log), build_log, NULL);
		printf ("Error in kernel: %s\n", build_log);
	}

	if (make_binary == true)
	{
		// Append file name to path
		std::string bin_path;
		bin_path.append(kernel_path);
		bin_path.append("bruteforce_cl.bin");

		// The following code is taken from this link: https://community.amd.com/thread/159033
		int err;
		err = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(size_t), &ret_num_devices, NULL);// Return 1 devices
		size_t *np = new size_t[ret_num_devices];//Create size array
		err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t)*ret_num_devices, np, NULL);//Load in np the size of binary
		char** bn = new char* [ret_num_devices]; //Create the binary array
		for(int i =0; i < ret_num_devices;i++)
		{
			bn[i] = new char[np[i]];
			printf("size = %d\n", np[i]);
		}
		err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char *)*ret_num_devices, bn, NULL); //Load the binary itself
		printf("%s\n", bn[0]); //Print the first binary.
		FILE *fop = fopen(bin_path.c_str(), "wb");
		fwrite(bn[0], sizeof(char), np[0], fop); // Save the binary
		fclose(fop);
		make_binary = false;
	}
}

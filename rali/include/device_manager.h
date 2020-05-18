#pragma once

#include <map>
#include <CL/cl.h>
#include <vx_ext_amd.h>
#include <VX/vx_types.h>
#include <memory>
#include "device_data_transfer_code.h"
struct DeviceResources {
    cl_context context;
    cl_device_id device_id;
    cl_command_queue cmd_queue;
    DeviceResources() { cmd_queue = nullptr; context = nullptr; device_id = nullptr; }
};


class CLProgram {
public:
    CLProgram(const DeviceResources* ocl, const DeviceCode& ocl_code): m_ocl(ocl), m_code(ocl_code) {}

    ~CLProgram();

    cl_int runKernel(const std::string& kernel_name, const std::vector<void*>&  args, const std::vector<size_t>& argSize, const std::vector<size_t>& globalWorkSize, const std::vector<size_t>& localWorkSize);

    cl_int buildAll();

    const cl_kernel& operator[](const std::string& kernel_name) const ;

    std::string getProgramName();

private:
    const DeviceResources* m_ocl;

    const DeviceCode& m_code;

    cl_program m_prog;

    std::map<std::string, cl_kernel> m_kernels;

};


class DeviceManager {
public:
    DeviceManager(){};

    cl_int initialize();
    
    DeviceResources resources();

    const CLProgram& operator[](const std::string& prog_name);

    void init_ocl(vx_context context);

    ~DeviceManager();

private:

    DeviceResources _resources;

    std::map<std::string, CLProgram> m_programs;
};

using pRaliOCL = std::shared_ptr<DeviceManager>;

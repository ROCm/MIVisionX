#pragma once

#include <map>
#include <CL/cl.h>
#include <memory>
#include "device_utility_code.h"

struct OCLResources {
    cl_command_queue cmd_queue;
    cl_context context;
    cl_device_id device_id;
    OCLResources( cl_context _context, cl_device_id _device_id, cl_command_queue _queue): cmd_queue (_queue), context(_context), device_id(_device_id) {}
    OCLResources() {cmd_queue = nullptr; context = nullptr; device_id = nullptr; }
};


class CLProgram {
public:
    CLProgram(const OCLResources* ocl, const DeviceCode& ocl_code): m_ocl(ocl), m_code(ocl_code) {}

    cl_int runKernel(const std::string& kernel_name, const std::vector<void*>&  args, const std::vector<size_t>& argSize, const std::vector<size_t>& globalWorkSize, const std::vector<size_t>& localWorkSize);

    cl_int buildAll();

    const cl_kernel& operator[](const std::string& kernel_name) const ;

    std::string getProgramName();

private:
    const OCLResources* m_ocl;

    const DeviceCode& m_code;

    cl_program m_prog;

    std::map<std::string, cl_kernel> m_kernels;

};


class DeviceManager {
public:
    DeviceManager(){};

    DeviceManager(cl_command_queue queue, cl_context context, cl_device_id device): _resources(context, device, queue) {}

    void set_resources(cl_command_queue queue, cl_context context, cl_device_id device) 
    {
        _resources.cmd_queue = queue;
        _resources.context = context;
        _resources.device_id = device;
    };

    cl_int initialize();
    
    OCLResources resources();

    const CLProgram& operator[](const std::string& prog_name);


    ~DeviceManager();

private:

    OCLResources _resources;

    std::map<std::string, CLProgram> m_programs;
};

using pRaliOCL = std::shared_ptr<DeviceManager>;

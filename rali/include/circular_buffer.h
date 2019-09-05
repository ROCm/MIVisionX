#pragma once
#include <vector>
#include <condition_variable>
#include <CL/cl.h>
#include "device_manager.h"
#include "commons.h"

enum class CIRCULAR_BUFFER_STATUS {
    OK = 0,
    BUFFER_TOO_SHALLOW,
    OPENCL_BUFFER_ALLOCATION_FAILED,
    OPENCL_MAP_BUFFER_FAILED,
    OCL_INFO_MISSING
};

class CircularBuffer {
public:
    CircularBuffer( OCLResources ocl, size_t buffer_depth );
    ~CircularBuffer();
    CIRCULAR_BUFFER_STATUS init(RaliMemType output_mem_type, size_t output_mem_size);
    CIRCULAR_BUFFER_STATUS sync();
    void cancel_reading();
    void cancel_writing();
    void done_writing();
    void done_reading();
    cl_mem get_read_buffer_dev();
    unsigned char* get_read_buffer_host();
    unsigned char*  get_write_buffer();
    size_t level();
private:
    void wait_if_empty();
    void wait_if_full();
    void increment_read_ptr();
    void increment_write_ptr();
    bool full();
    bool empty();    
    const size_t BUFF_DEPTH;

    /*
     *  Pinned memory allocated on the host used for fast host to device memory transactions,
     *  or the regular host memory buffers in the CPU affinity case.
     */
    std::vector<cl_mem> _dev_buffer;// Actual memory allocated on the device (in the case of GPU affinity)
    std::vector<unsigned char*> _host_buffer;
    std::condition_variable _wait_for_load;
    std::condition_variable _wait_for_unload;
    std::mutex _lock;
    RaliMemType _output_mem_type;
    size_t _output_mem_size;

    size_t _write_ptr;
    size_t _read_ptr;
    size_t _level;
    cl_command_queue _cl_cmdq = nullptr;
    cl_context _cl_context = nullptr;
    cl_device_id _device_id = nullptr;
};
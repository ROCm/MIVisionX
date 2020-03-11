#pragma once
#include <vector>
#include <condition_variable>
#include <CL/cl.h>
#include "device_manager.h"
#include "commons.h"

class CircularBuffer {
public:
    CircularBuffer(DeviceResources ocl, size_t buffer_depth );
    ~CircularBuffer();
    void init(RaliMemType output_mem_type, size_t output_mem_size);
    void sync();
    void cancel_reading();
    void cancel_writing();
    void push();
    void pop();
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
    cl_command_queue _cl_cmdq = nullptr;
    cl_context _cl_context = nullptr;
    cl_device_id _device_id = nullptr;
    std::vector<cl_mem> _dev_buffer;// Actual memory allocated on the device (in the case of GPU affinity)
    std::vector<unsigned char*> _host_buffer_ptrs;
    std::vector<std::vector<unsigned char>> _actual_host_buffers;
    std::condition_variable _wait_for_load;
    std::condition_variable _wait_for_unload;
    std::mutex _lock;
    RaliMemType _output_mem_type;
    size_t _output_mem_size;

    size_t _write_ptr;
    size_t _read_ptr;
    size_t _level;
};
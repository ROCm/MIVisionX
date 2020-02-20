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
    void sync();// Syncs device buffers with host
    void unblock_reader();// Unblocks the thread currently waiting on a call to get_read_buffer
    void unblock_writer();// Unblocks the thread currently waiting on get_write_buffer
    void push();// The latest write goes through, effectively adds one element to the buffer
    void pop();// The oldest write will be erased and overwritten in upcoming writes
    cl_mem get_read_buffer_dev();
    unsigned char* get_read_buffer_host();// blocks the caller if the buffer is empty
    unsigned char*  get_write_buffer(); // blocks the caller if the buffer is full
    size_t level();// Returns the number of elements stored
    void reset();// sets the buffer level to 0
    void block_if_empty();// blocks the caller if the buffer is empty
    void block_if_full();// blocks the caller if the buffer is full

private:
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
    bool _initialized = false;
    size_t _write_ptr;
    size_t _read_ptr;
    size_t _level;
};
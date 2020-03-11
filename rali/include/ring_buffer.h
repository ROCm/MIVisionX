#pragma once
#include "commons.h"
#include <vector>
#include <condition_variable>
#include <CL/cl.h>
#include "device_manager.h"
#include "commons.h"
class RingBuffer
{
public:
    explicit RingBuffer(unsigned buffer_depth);
    ~RingBuffer();
    size_t level();
    void init(RaliMemType mem_type, DeviceResources dev, unsigned buffer_size, unsigned sub_buffer_count);
    std::vector<void*> get_read_buffers() ;
    void* get_host_master_read_buffer();
    std::vector<void*> get_write_buffers();
    void pop();
    void push();
    void cancel_reading()
    {
        // Wake up the reader thread in case it's waiting for a load
        _wait_for_load.notify_one();
    }

    void cancel_writing()
    {
        // Wake up the writer thread in case it's waiting for an unload
        _wait_for_unload.notify_one();
    }
    RaliMemType mem_type() { return _mem_type; }
private:
    void wait_if_empty();
    void wait_if_full();
    void increment_read_ptr();
    void increment_write_ptr();
    bool full();
    bool empty();
    const unsigned BUFF_DEPTH;
    unsigned _sub_buffer_size;
    unsigned _sub_buffer_count;
    std::mutex _lock;
    std::condition_variable _wait_for_load;
    std::condition_variable _wait_for_unload;
    std::vector<std::vector<void*>> _dev_sub_buffer;
    std::vector<std::vector<unsigned char>> _host_master_buffers;
    std::vector<std::vector<void*>> _host_sub_buffers;

    RaliMemType _mem_type;
    DeviceResources _dev;
    size_t _write_ptr;
    size_t _read_ptr;
    size_t _level;

};

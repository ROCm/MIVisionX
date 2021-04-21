/*
Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.

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

#pragma once
#include <vector>
#include <condition_variable>
#include <CL/cl.h>
#include <queue>
#include "device_manager.h"
#include "commons.h"
struct decoded_image_info
{
    std::vector<std::string> _image_names;
    std::vector<uint32_t> _roi_width;
    std::vector<uint32_t> _roi_height;
    std::vector<uint32_t> _original_width;
    std::vector<uint32_t> _original_height;
};

class CircularBuffer
{
public:
    CircularBuffer(DeviceResources ocl, size_t buffer_depth );
    ~CircularBuffer();
    void init(RaliMemType output_mem_type, size_t output_mem_size);
    void sync();// Syncs device buffers with host
    void unblock_reader();// Unblocks the thread currently waiting on a call to get_read_buffer
    void unblock_writer();// Unblocks the thread currently waiting on get_write_buffer
    void push();// The latest write goes through, effectively adds one element to the buffer
    void pop();// The oldest write will be erased and overwritten in upcoming writes
    void set_image_info(const decoded_image_info& info) { _last_image_info = info; }
    decoded_image_info& get_image_info();
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
    decoded_image_info _last_image_info;
    std::queue<decoded_image_info> _circ_image_info;//!< Stores the loaded images names, decoded_width and decoded_height(data is stored in the _circ_buff)
    std::mutex _names_buff_lock;
    /*
     *  Pinned memory allocated on the host used for fast host to device memory transactions,
     *  or the regular host memory buffers in the host processing case.
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
    const size_t MEM_ALIGNMENT = 256;
    size_t _write_ptr;
    size_t _read_ptr;
    size_t _level;
};
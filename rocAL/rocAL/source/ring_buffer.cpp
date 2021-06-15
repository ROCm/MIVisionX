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

#include <device_manager.h>
#include "ring_buffer.h"

RingBuffer::RingBuffer(unsigned buffer_depth):
        BUFF_DEPTH(buffer_depth),
        _dev_sub_buffer(buffer_depth),
        _host_master_buffers(BUFF_DEPTH)
{
    reset();
}
void RingBuffer::block_if_empty()
{
    std::unique_lock<std::mutex> lock(_lock);
    if(empty())
    { // if the current read buffer is being written wait on it
        if(_dont_block)
            return;
        _wait_for_load.wait(lock);
    }
}

void RingBuffer:: block_if_full()
{
    std::unique_lock<std::mutex> lock(_lock);
    // Write the whole buffer except for the last spot which is being read by the reader thread
    if(full())
    {
        if(_dont_block)
            return;
        _wait_for_unload.wait(lock);
    }
}
std::vector<void*> RingBuffer::get_read_buffers()
{
    block_if_empty();
    if(_mem_type == RaliMemType::OCL)
        return _dev_sub_buffer[_read_ptr];

    return _host_sub_buffers[_read_ptr];
}

void *RingBuffer::get_host_master_read_buffer() {
    block_if_empty();
    if(_mem_type == RaliMemType::OCL)
        return nullptr;

    return _host_master_buffers[_read_ptr];
}


std::vector<void*> RingBuffer::get_write_buffers()
{
    block_if_full();
    if(_mem_type == RaliMemType::OCL)
        return _dev_sub_buffer[_write_ptr];

    return _host_sub_buffers[_write_ptr];
}


void RingBuffer::unblock_reader()
{
    // Wake up the reader thread in case it's waiting for a load
    _wait_for_load.notify_all();
}

void RingBuffer::release_all_blocked_calls()
{
    _dont_block = true;
    unblock_reader();
    unblock_writer();
}

void RingBuffer::release_if_empty()
{
    if (empty()) unblock_reader();
}

void RingBuffer::unblock_writer()
{
    // Wake up the writer thread in case it's waiting for an unload
    _wait_for_unload.notify_all();
}
void RingBuffer::init(RaliMemType mem_type, DeviceResources dev, unsigned sub_buffer_size, unsigned sub_buffer_count)
{
    _mem_type = mem_type;
    _dev = dev;
    _sub_buffer_size = sub_buffer_size;
    _sub_buffer_count = sub_buffer_count;
    if(BUFF_DEPTH < 2)
        THROW ("Error internal buffer size for the ring buffer should be greater than one")

    // Allocating buffers
    if(mem_type== RaliMemType::OCL)
    {
        if(_dev.cmd_queue == nullptr || _dev.device_id == nullptr || _dev.context == nullptr)
            THROW("Error ocl structure needed since memory type is OCL");

        cl_int err = CL_SUCCESS;

        for(size_t buffIdx = 0; buffIdx < BUFF_DEPTH; buffIdx++)
        {
            cl_mem_flags flags = CL_MEM_READ_ONLY;

            _dev_sub_buffer[buffIdx].resize(_sub_buffer_count);
            for(unsigned sub_idx = 0; sub_idx < _sub_buffer_count; sub_idx++)
            {
                _dev_sub_buffer[buffIdx][sub_idx] =  clCreateBuffer(_dev.context, flags, sub_buffer_size, NULL, &err);

                if(err)
                {
                    _dev_sub_buffer.clear();
                    THROW("clCreateBuffer of size " + TOSTR(sub_buffer_size) + " index " + TOSTR(sub_idx) +
                          " failed " + TOSTR(err));
                }

                clRetainMemObject((cl_mem)_dev_sub_buffer[buffIdx][sub_idx]);
            }

        }
    }
    else
    {
        _host_sub_buffers.resize(BUFF_DEPTH);
        for(size_t buffIdx = 0; buffIdx < BUFF_DEPTH; buffIdx++)
        {
            const size_t master_buffer_size = sub_buffer_size * sub_buffer_count;
            // a minimum of extra MEM_ALIGNMENT is allocated
            _host_master_buffers[buffIdx] = aligned_alloc(MEM_ALIGNMENT, MEM_ALIGNMENT * (master_buffer_size / MEM_ALIGNMENT + 1));
            _host_sub_buffers[buffIdx].resize(_sub_buffer_count);
            for(size_t sub_buff_idx = 0; sub_buff_idx < _sub_buffer_count; sub_buff_idx++)
                _host_sub_buffers[buffIdx][sub_buff_idx] = (unsigned char*)_host_master_buffers[buffIdx] + _sub_buffer_size * sub_buff_idx;
        }
    }
}
void RingBuffer::push()
{
    // pushing and popping to and from image and metadata buffer should be atomic so that their level stays the same at all times
    std::unique_lock<std::mutex> lock(_names_buff_lock);
    _meta_ring_buffer.push(_last_image_meta_data);
    increment_write_ptr();
}

void RingBuffer::pop()
{
    if(empty())
        return;
    // pushing and popping to and from image and metadata buffer should be atomic so that their level stays the same at all times
    std::unique_lock<std::mutex> lock(_names_buff_lock);
    increment_read_ptr();
    _meta_ring_buffer.pop();
}

void RingBuffer::reset()
{
    _write_ptr = 0;
    _read_ptr = 0;
    _level = 0;
    _dont_block = false;
    while(!_meta_ring_buffer.empty())
        _meta_ring_buffer.pop();
}

RingBuffer::~RingBuffer()
{
    //  if(_mem_type!= RaliMemType::OCL)
    //      return;
    if (_mem_type == RaliMemType::HOST) {
        for (unsigned idx = 0; idx < _host_master_buffers.size(); idx++)
            if (_host_master_buffers[idx]) {
                free(_host_master_buffers[idx]);
            }

        _host_master_buffers.clear();
        _host_sub_buffers.clear();
    }
    else if (_mem_type == RaliMemType::OCL) {
        for (size_t buffIdx = 0; buffIdx < _dev_sub_buffer.size(); buffIdx++)
            for (unsigned sub_buf_idx = 0; sub_buf_idx < _dev_sub_buffer[buffIdx].size(); sub_buf_idx++)
                if (_dev_sub_buffer[buffIdx][sub_buf_idx])
                    if (clReleaseMemObject((cl_mem) _dev_sub_buffer[buffIdx][sub_buf_idx]) != CL_SUCCESS)
                        ERR("Could not release ocl memory in the ring buffer")
        _dev_sub_buffer.clear();
    }
}

bool RingBuffer::empty()
{
    return (_level <= 0);
}

bool RingBuffer::full()
{
    return (_level >= BUFF_DEPTH - 1);
}

size_t RingBuffer::level()
{
    return _level;
}
void RingBuffer::increment_read_ptr()
{
    std::unique_lock<std::mutex> lock(_lock);
    _read_ptr = (_read_ptr+1)%BUFF_DEPTH;
    _level--;
    lock.unlock();
    // Wake up the writer thread (in case waiting) since there is an empty spot to write to,
    _wait_for_unload.notify_all();

}

void RingBuffer::increment_write_ptr()
{
    std::unique_lock<std::mutex> lock(_lock);
    _write_ptr = (_write_ptr+1)%BUFF_DEPTH;
    _level++;
    lock.unlock();
    // Wake up the reader thread (in case waiting) since there is a new load to be read
    _wait_for_load.notify_all();
}

void RingBuffer::set_meta_data( ImageNameBatch names, pMetaDataBatch meta_data)
{
    _last_image_meta_data = std::move(std::make_pair(std::move(names), meta_data));
}

MetaDataNamePair& RingBuffer::get_meta_data()
{
    block_if_empty();
    std::unique_lock<std::mutex> lock(_names_buff_lock);
    if(_level != _meta_ring_buffer.size())
        THROW("ring buffer internals error, image and metadata sizes not the same "+TOSTR(_level) + " != "+TOSTR(_meta_ring_buffer.size()))
    return  _meta_ring_buffer.front();
}


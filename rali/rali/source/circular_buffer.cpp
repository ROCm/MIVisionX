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

#include "circular_buffer.h"
#include "log.h"
CircularBuffer::CircularBuffer(DeviceResources ocl, size_t buffer_depth):
        BUFF_DEPTH(buffer_depth),
        _cl_cmdq(ocl.cmd_queue),
        _cl_context(ocl.context),
        _device_id(ocl.device_id),
        _dev_buffer(BUFF_DEPTH, nullptr),
        _host_buffer_ptrs(BUFF_DEPTH, nullptr),
        _write_ptr(0),
        _read_ptr(0),
        _level(0)
{
    for(size_t bufIdx = 0; bufIdx < BUFF_DEPTH; bufIdx++)
        _dev_buffer[bufIdx] = nullptr;

}

void CircularBuffer::reset()
{
    _write_ptr = 0;
    _read_ptr = 0;
    _level = 0;
    while(!_circ_image_info.empty())
        _circ_image_info.pop();
}

void CircularBuffer::unblock_reader()
{
    if(!_initialized)
        return;
    // Wake up the reader thread in case it's waiting for a load
    _wait_for_load.notify_one();
}

void CircularBuffer::unblock_writer()
{
    if(!_initialized)
        return;
    // Wake up the writer thread in case it's waiting for an unload
    _wait_for_unload.notify_one();
}


cl_mem CircularBuffer::get_read_buffer_dev()
{
    block_if_empty();
    return _dev_buffer[_read_ptr];
}

unsigned char* CircularBuffer::get_read_buffer_host()
{
    if(!_initialized)
        THROW("Circular buffer not initialized")
    block_if_empty();
    return _host_buffer_ptrs[_read_ptr];
}

unsigned char*  CircularBuffer::get_write_buffer()
{
    if(!_initialized)
        THROW("Circular buffer not initialized")
    block_if_full();
    return(_host_buffer_ptrs[_write_ptr]);
}

void CircularBuffer::sync()
{
    if(!_initialized)
        return;
    cl_int err = CL_SUCCESS;
    if(_output_mem_type== RaliMemType::OCL) 
    {
#if 0
        if(clEnqueueWriteBuffer(_cl_cmdq, _dev_sub_buffer[_write_ptr], CL_TRUE, 0, _output_mem_size, _host_buffer_ptrs[_write_ptr], 0, NULL, NULL) != CL_SUCCESS)
            THROW("clEnqueueMapBuffer of size "+ TOSTR(_output_mem_size) + " failed " + TOSTR(err));

#else        
        //NOTE: instead of calling clEnqueueWriteBuffer (shown above), 
        // an unmap/map cen be done to make sure data is copied from the host to device, it's fast
        //NOTE: Using clEnqueueUnmapMemObject/clEnqueuenmapMemObject when buffer is allocated with 
        // CL_MEM_ALLOC_HOST_PTR adds almost no overhead
        clEnqueueUnmapMemObject(_cl_cmdq, _dev_buffer[_write_ptr], _host_buffer_ptrs[_write_ptr], 0, NULL, NULL);
        _host_buffer_ptrs[_write_ptr] = (unsigned char*) clEnqueueMapBuffer(_cl_cmdq,
                                                                            _dev_buffer[_write_ptr] ,
                                                                            CL_FALSE,
                                                                            CL_MAP_WRITE,
                                                                            0,
                                                                            _output_mem_size,
                                                                            0, NULL, NULL, &err );
        if(err)
            THROW("clEnqueueUnmapMemObject of size "+ TOSTR(_output_mem_size) + " failed " + TOSTR(err));

#endif        
    }
    else 
    {
        // For the host processing no copy is needed, since data is already loaded in the host buffers
        // and handle will be swaped on it
    }
}

void CircularBuffer::push()
{
    if(!_initialized)
        return;
    sync();
    // Pushing to the _circ_buff and _circ_buff_names must happen all at the same time
    std::unique_lock<std::mutex> lock(_names_buff_lock);
    _circ_image_info.push(_last_image_info);
    increment_write_ptr();
}

void CircularBuffer::pop()
{
    if(!_initialized)
        return;
    // Pushing to the _circ_buff and _circ_buff_names must happen all at the same time
    std::unique_lock<std::mutex> lock(_names_buff_lock);
    increment_read_ptr();
    _circ_image_info.pop();
}
void CircularBuffer::init(RaliMemType output_mem_type, size_t output_mem_size)
{
    if(_initialized)
        return;
    _output_mem_type = output_mem_type;
    _output_mem_size = output_mem_size;
    if(BUFF_DEPTH < 2)
        THROW ("Error internal buffer size for the circular buffer should be greater than one")
    
    // Allocating buffers
    if(_output_mem_type== RaliMemType::OCL) 
    {
        if(_cl_cmdq == nullptr || _device_id == nullptr || _cl_context == nullptr)
            THROW("Error ocl structure needed since memory type is OCL");

        cl_int err = CL_SUCCESS;

        for(size_t buffIdx = 0; buffIdx < BUFF_DEPTH; buffIdx++)
        {
            //NOTE: we don't need to use CL_MEM_ALLOC_HOST_PTR memory if this buffer is not going to be
            // used in the host. But we cannot ensure which Rali's copy function is going to be called 
            // (copy to host or OCL) by the user
            _dev_buffer[buffIdx] = clCreateBuffer(  _cl_context, 
                                                    CL_MEM_READ_ONLY|CL_MEM_ALLOC_HOST_PTR,
                                                    _output_mem_size, NULL, &err);// Create pinned memory
            if (!_dev_buffer[buffIdx]  || err)
                THROW("clCreateBuffer of size" + TOSTR(_output_mem_size)+  "failed " + TOSTR(err));

            //TODO: we don't need to map the buffers to host here if the output of the output of this
            //  loader_module is not required by the user to be part of the augmented output
            _host_buffer_ptrs[buffIdx] = (unsigned char*) clEnqueueMapBuffer(_cl_cmdq,
                                                                             _dev_buffer[buffIdx] ,
                                                                             CL_TRUE, CL_MAP_WRITE,
                                                                             0,
                                                                             _output_mem_size,
                                                                             0, NULL, NULL, &err );
            if(err)
                THROW("clEnqueueMapBuffer of size" + TOSTR(_output_mem_size)+  "failed " + TOSTR(err));
            clRetainMemObject(_dev_buffer[buffIdx]);


        }
    } 
    else 
    {
        for(size_t buffIdx = 0; buffIdx < BUFF_DEPTH; buffIdx++)
        {
            // a minimum of extra MEM_ALIGNMENT is allocated
            _host_buffer_ptrs[buffIdx] = (unsigned char*)aligned_alloc(MEM_ALIGNMENT, MEM_ALIGNMENT * (_output_mem_size / MEM_ALIGNMENT + 1));
        }


    }
    _initialized = true;
}

bool CircularBuffer::empty()
{
    return (_level <= 0);
}

bool CircularBuffer::full()
{
    return (_level >= BUFF_DEPTH - 1);
}

size_t CircularBuffer::level()
{
    return _level;
}

void CircularBuffer::increment_read_ptr()
{
    std::unique_lock<std::mutex> lock(_lock);
    _read_ptr = (_read_ptr+1)%BUFF_DEPTH;
    _level--;
    lock.unlock();
    // Wake up the writer thread (in case waiting) since there is an empty spot to write to,
    _wait_for_unload.notify_all();

}

void CircularBuffer::increment_write_ptr()
{
    std::unique_lock<std::mutex> lock(_lock);
    _write_ptr = (_write_ptr+1)%BUFF_DEPTH;
    _level++;
    lock.unlock();
    // Wake up the reader thread (in case waiting) since there is a new load to be read
    _wait_for_load.notify_all();
}

void CircularBuffer::block_if_empty()
{
    std::unique_lock<std::mutex> lock(_lock);
    if(empty()) 
    { // if the current read buffer is being written wait on it
        _wait_for_load.wait(lock);
    }
}

void CircularBuffer:: block_if_full()
{
    std::unique_lock<std::mutex> lock(_lock);
    // Write the whole buffer except for the last spot which is being read by the reader thread
    if(full()) 
    {
        _wait_for_unload.wait(lock);
    }
}

CircularBuffer::~CircularBuffer()
{
    for(size_t buffIdx = 0; buffIdx < BUFF_DEPTH; buffIdx++) 
    {
        if(_output_mem_type== RaliMemType::OCL) 
        {
            if(clEnqueueUnmapMemObject(_cl_cmdq, _dev_buffer[buffIdx], _host_buffer_ptrs[buffIdx], 0, NULL, NULL) != CL_SUCCESS)
                ERR("Could not unmap ocl memory")
            if(clReleaseMemObject(_dev_buffer[buffIdx]) != CL_SUCCESS)
                ERR("Could not release ocl memory in the circular buffer")
        }
        else
        {
            free(_host_buffer_ptrs[buffIdx]);
        }
    }

    _dev_buffer.clear();
    _host_buffer_ptrs.clear();
    _write_ptr = 0;
    _read_ptr = 0;
    _level = 0;
    _cl_cmdq = 0;
    _cl_context = 0;
    _device_id = 0;
    _initialized = false;
}

decoded_image_info &CircularBuffer::get_image_info()
{
    block_if_empty();
    std::unique_lock<std::mutex> lock(_names_buff_lock);
    if(_level != _circ_image_info.size())
        THROW("CircularBuffer internals error, image and image info sizes not the same "+TOSTR(_level) + " != "+TOSTR(_circ_image_info.size()))
    return  _circ_image_info.front();
}



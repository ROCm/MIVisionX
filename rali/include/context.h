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

#include "commons.h"
#include "graph.h"
#include "master_graph.h"


struct Context
{
    explicit Context(size_t batch_size, RaliAffinity affinity, int gpu_id , size_t cpu_thread_count ):
    affinity(affinity),
    _user_batch_size(batch_size)
    {
        LOG("Processing on " + STR(((affinity == RaliAffinity::CPU)?" CPU": " GPU")))
        master_graph = std::make_shared<MasterGraph>(batch_size, affinity, gpu_id, cpu_thread_count);
        _internal_batch_size = master_graph->internal_batch_size();
    }
    ~Context()
    {
        clear_errors();
    };
    std::shared_ptr<MasterGraph> master_graph;

    RaliAffinity affinity;
    bool no_error() { return error.empty(); }
    const char* error_msg() { return error.c_str(); }
    void capture_error(const std::string& err_msg) { error = err_msg; }
    Timing timing()
    {
        return master_graph->timing();
    }
    size_t user_batch_size() { return _user_batch_size; }
    size_t internal_batch_size() { return _internal_batch_size; }
private:
    void clear_errors() { error = "";}
    std::string error;
    size_t _user_batch_size;
    size_t _internal_batch_size;
};
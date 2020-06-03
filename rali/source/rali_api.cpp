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

#include <string>
#include <exception>
#include "commons.h"
#include "context.h"
#include "rali_api.h"

RaliStatus RALI_API_CALL
raliRelease(RaliContext p_context)
{
    // Deleting context is required to call the destructor of all the member objects
    auto context = static_cast<Context*>(p_context);
    delete context;
    return RALI_OK;
}
RaliContext RALI_API_CALL
raliCreate(
        size_t batch_size,
        RaliProcessMode affinity,
        int gpu_id,
        size_t cpu_thread_count)
{
    RaliContext context = nullptr;
    try
    {
        auto translate_process_mode = [](RaliProcessMode process_mode)
        {
            switch(process_mode)
            {
                case RALI_PROCESS_GPU:
                    return RaliAffinity::GPU;
                case RALI_PROCESS_CPU:
                    return RaliAffinity::CPU;
                default:
                    THROW("Unkown Rali process mode")
            }
        };
        context = new Context(batch_size, translate_process_mode(affinity), gpu_id, cpu_thread_count);
        // Reset seed in case it's being randomized during context creation
    }
    catch(const std::exception& e)
    {
        ERR( STR("Failed to init the Rali context, ") + STR(e.what()))
    }
    return context;
}

RaliStatus RALI_API_CALL
raliRun(RaliContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    try
    {
        auto ret = context->master_graph->run();
        if(ret != MasterGraph::Status::OK)
            return RALI_RUNTIME_ERROR;
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
        return RALI_RUNTIME_ERROR;
    }
    return RALI_OK;
}

RaliStatus RALI_API_CALL
raliVerify(RaliContext p_context)
{
    auto context = static_cast<Context*>(p_context);
    try
    {
        context->master_graph->build();
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
        return RALI_RUNTIME_ERROR;
    }
    return RALI_OK;
}





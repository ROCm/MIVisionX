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

#include "commons.h"
#include "context.h"
#include "rali_api.h"
#include "CL/cl.h"

RaliStatus RALI_API_CALL
raliCopyToOutputTensor32(RaliContext p_context, float *out_ptr, RaliTensorLayout tensor_format, float multiplier0,
                       float multiplier1, float multiplier2, float offset0, float offset1, float offset2,
                       bool reverse_channels)
{
    auto context = static_cast<Context*>(p_context);
    try
    {
        auto tensor_layout = (tensor_format == RALI_NHWC) ?  RaliTensorFormat::NHWC : RaliTensorFormat::NCHW;
        //auto tensor_output_data_type = (tensor_data_type == RALI_FP32) ? RaliTensorDataType::FP32 : RaliTensorDataType::FP16;
        context->master_graph->copy_out_tensor(out_ptr, tensor_layout, multiplier0, multiplier1, multiplier2,
                offset0, offset1, offset2, reverse_channels, RaliTensorDataType::FP32);
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
raliCopyToOutputTensor16(RaliContext p_context, half *out_ptr, RaliTensorLayout tensor_format, float multiplier0,
                       float multiplier1, float multiplier2, float offset0, float offset1, float offset2,
                       bool reverse_channels)
{
    auto context = static_cast<Context*>(p_context);
    try
    {
        auto tensor_layout = (tensor_format == RALI_NHWC) ?  RaliTensorFormat::NHWC : RaliTensorFormat::NCHW;
        //auto tensor_output_data_type = (tensor_data_type == RALI_FP32) ? RaliTensorDataType::FP32 : RaliTensorDataType::FP16;
        context->master_graph->copy_out_tensor(out_ptr, tensor_layout, multiplier0, multiplier1, multiplier2,
                offset0, offset1, offset2, reverse_channels, RaliTensorDataType::FP16);
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
raliCopyToOutput(
        RaliContext p_context,
        cl_mem out_ptr,
        size_t out_size)
{
    auto context = static_cast<Context*>(p_context);
    try
    {
        context->master_graph->copy_output(out_ptr, out_size);
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
raliCopyToOutput(
        RaliContext p_context,
        unsigned char * out_ptr,
        size_t out_size)
{
    auto context = static_cast<Context*>(p_context);
    try
    {
        context->master_graph->copy_output(out_ptr);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
        return RALI_RUNTIME_ERROR;
    }
    return RALI_OK;
}


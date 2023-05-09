/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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
#include "rocal_api.h"
#if ENABLE_OPENCL
#include "CL/cl.h"
#endif

RocalStatus ROCAL_API_CALL
rocalToTensor32(RocalContext p_context, float *out_ptr, RocalTensorLayout tensor_format, float multiplier0,
                       float multiplier1, float multiplier2, float offset0, float offset1, float offset2,
                       bool reverse_channels, RocalOutputMemType output_mem_type)
{
    auto context = static_cast<Context*>(p_context);
    try
    {
        auto tensor_layout = (tensor_format == ROCAL_NHWC) ?  RocalTensorFormat::NHWC : RocalTensorFormat::NCHW;
        //auto tensor_output_data_type = (tensor_data_type == ROCAL_FP32) ? RocalTensorDataType::FP32 : RocalTensorDataType::FP16;
        context->master_graph->to_tensor(out_ptr, tensor_layout, multiplier0, multiplier1, multiplier2,
                offset0, offset1, offset2, reverse_channels, RocalTensorDataType::FP32, output_mem_type);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
        return ROCAL_RUNTIME_ERROR;
    }
    return ROCAL_OK;
}

RocalStatus ROCAL_API_CALL
rocalToTensor16(RocalContext p_context, half *out_ptr, RocalTensorLayout tensor_format, float multiplier0,
                       float multiplier1, float multiplier2, float offset0, float offset1, float offset2,
                       bool reverse_channels, RocalOutputMemType output_mem_type)
{
    auto context = static_cast<Context*>(p_context);
    try
    {
        auto tensor_layout = (tensor_format == ROCAL_NHWC) ?  RocalTensorFormat::NHWC : RocalTensorFormat::NCHW;
        //auto tensor_output_data_type = (tensor_data_type == ROCAL_FP32) ? RocalTensorDataType::FP32 : RocalTensorDataType::FP16;
        context->master_graph->to_tensor(out_ptr, tensor_layout, multiplier0, multiplier1, multiplier2,
                offset0, offset1, offset2, reverse_channels, RocalTensorDataType::FP16, output_mem_type);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
        return ROCAL_RUNTIME_ERROR;
    }
    return ROCAL_OK;
}

RocalStatus ROCAL_API_CALL
rocalToTensor(RocalContext p_context, void *out_ptr, RocalTensorLayout tensor_format, RocalTensorOutputType tensor_output_type, float multiplier0,
                       float multiplier1, float multiplier2, float offset0, float offset1, float offset2,
                       bool reverse_channels, RocalOutputMemType output_mem_type)
{
    auto context = static_cast<Context*>(p_context);
    try
    {
        auto tensor_layout = (tensor_format == ROCAL_NHWC) ?  RocalTensorFormat::NHWC : RocalTensorFormat::NCHW;
        auto tensor_output_data_type = (tensor_output_type == ROCAL_FP32) ? RocalTensorDataType::FP32 : RocalTensorDataType::FP16;
        context->master_graph->to_tensor(out_ptr, tensor_layout, multiplier0, multiplier1, multiplier2,
                offset0, offset1, offset2, reverse_channels, tensor_output_data_type, output_mem_type);
    }
    catch(const std::exception& e)
    {
        context->capture_error(e.what());
        ERR(e.what())
        return ROCAL_RUNTIME_ERROR;
    }
    return ROCAL_OK;
}


RocalStatus ROCAL_API_CALL
rocalCopyToOutput(
        RocalContext p_context,
        unsigned char * out_ptr,
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
        return ROCAL_RUNTIME_ERROR;
    }
    return ROCAL_OK;
}

void
ROCAL_API_CALL rocalSetOutputs(RocalContext p_context, unsigned int num_of_outputs, std::vector<RocalImage> &output_images)
{
    if (!p_context)
        THROW("Invalid rocal context passed to rocalSetOutputs")
    auto context = static_cast<Context *>(p_context);
    std::vector<Image*> output_images_vector ;
    for (auto& it : output_images) {
        auto img = static_cast<Image*>(it);
        context->master_graph->set_output(img);
    }
}



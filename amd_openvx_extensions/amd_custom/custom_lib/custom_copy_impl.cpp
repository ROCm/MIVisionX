/*
Copyright (c) 2017 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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
#include "custom_copy_impl.h"
#include <stdio.h>
#include <stdlib.h>
#include "hip/hip_runtime_api.h"
#include "hip/hip_runtime.h"


customStatus_t customCopy::Setup(customTensorDesc &inputdesc, customTensorDesc &outputdesc, customBackend backend, customStream stream):
            _input_desc(inputdesc), _output_desc(outputdesc), _backend(backend), _stream(stream)
{
    if ((_input_desc.data_type != _output_desc.data_type) || (_input_desc.dims[0] != _output_desc.dims[0]) || 
        (_input_desc.dims[1] != _output_desc.dims[1]) || (_input_desc.dims[2] != _output_desc.dims[2]) || (_input_desc.dims[3] != _output_desc.dims[3]))
        return  customStatus_t::customStatusInvalidValue;
    return customStatusSuccess;
}

customStatus_t customCopy::Execute(void *input_handle, customTensorDesc &inputdesc, void *output_handle, customTensorDesc &outputdesc)
{
    unsigned size = outputdesc.dims[0] * outputdesc.dims[1] * outputdesc.dims[3];
    unsigned batch_size = outputdesc.dims[3];

    if (_backend == CPU)
    {
    #pragma omp parallel for num_threads(batch_size)
        unsigned char *src, *dst;
        for (size_t i = 0; i < batch_size; i++) {
            src = (unsigned char *)input_handle + size*i;
            dst = (unsigned char *)output_handle + size*i;
            memcpy(dst, src, size);
        }
    }else{
        for (size_t i = 0; i < batch_size; i++) {
            src = (unsigned char *)input_handle + size*i;
            dst = (unsigned char *)output_handle + size*i;
            hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice);
        }      
    }
    return customStatusSuccess;
}

customStatus_t customCopy::Shutdown(void *input_handle, customTensorDesc &inputdesc, void *output_handle, customTensorDesc &outputdesc, customBackend backend, customStream stream)
{
    // nothing to do since we don't have any local resources to release.
    return customStatusSuccess;
}
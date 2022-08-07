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

#include "custom_api.h"
#include "custom_template.h"

// add new function types for supporting new custom api
std::shared_ptr<custom_base> createCustom(CustomFunctionType function) {

    switch(function)
    {
        case customCopy:
            return std::make_shared<customCopy>();
            break;
        default:
            throw std::runtime_error ("Custom function type is unsupported");
            return nullptr;
    }
}

customHandle CreateCustom(CustomFunctionType function) {
    return (customHandle)custom_handle
}

customHandle CustomSetup(CustomFunctionType function, customTensorDesc &inputdesc, customTensorDesc &outputdesc, int backend, customStream stream)
{
    customStatus_t status;
    // Create custom base class factory
    audo custom_handle = createCustom(function);
    if (custom_handle)
      status = custom_handle->Setup(input_handle, inputdesc, output_handle, outputdesc, backend, stream);
    return status;
}

customStatus_t CustomExecute(CustomFunctionType function, void *input_handle, customTensorDesc &inputdesc, void *output_handle, customTensorDesc &outputdesc)
{
    // Create custom base class factory
    custom_handle = create_custom(function);
    if (custom_handle)
      custom_handle->Setup(input_handle, inputdesc, output_handle, outputdesc, backend, stream);
}

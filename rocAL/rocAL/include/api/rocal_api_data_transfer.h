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

#ifndef MIVISIONX_ROCAL_API_DATA_TRANSFER_H
#define MIVISIONX_ROCAL_API_DATA_TRANSFER_H
#include "rocal_api_types.h"

extern "C"  RocalStatus   ROCAL_API_CALL rocalCopyToOutput(RocalContext context, unsigned char * out_ptr, size_t out_size);

extern "C"  RocalStatus   ROCAL_API_CALL rocalToTensor32(RocalContext rocal_context, float *out_ptr,
                                                              RocalTensorLayout tensor_format, float multiplier0,
                                                              float multiplier1, float multiplier2, float offset0,
                                                              float offset1, float offset2,
                                                              bool reverse_channels, RocalOutputMemType output_mem_type);

extern "C"  RocalStatus   ROCAL_API_CALL rocalToTensor16(RocalContext rocal_context, half *out_ptr,
                                                              RocalTensorLayout tensor_format, float multiplier0,
                                                              float multiplier1, float multiplier2, float offset0,
                                                              float offset1, float offset2,
                                                              bool reverse_channels, RocalOutputMemType output_mem_type);

extern "C"  RocalStatus   ROCAL_API_CALL rocalToTensor(RocalContext rocal_context, void *out_ptr,
                                                              RocalTensorLayout tensor_format, RocalTensorOutputType tensor_output_type,
                                                              float multiplier0, float multiplier1, float multiplier2, float offset0,
                                                              float offset1, float offset2,
                                                              bool reverse_channels, RocalOutputMemType output_mem_type);
///
/// \param rocal_context
/// \param output_images The buffer that will be filled with output images with set_output = True
extern "C" void ROCAL_API_CALL rocalSetOutputs(RocalContext p_context, unsigned int num_of_outputs, std::vector<RocalImage> &output_images);


#endif //MIVISIONX_ROCAL_API_DATA_TRANSFER_H

/*
MIT License
Copyright (c) 2019 - 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef ROCAL_H
#define ROCAL_H

#include "rocal_api_types.h"
#include "rocal_api_parameters.h"
#include "rocal_api_data_loaders.h"
#include "rocal_api_augmentation.h"
#include "rocal_api_data_transfer.h"
#include "rocal_api_meta_data.h"
#include "rocal_api_info.h"

/// Creates the context for a new augmentation pipeline. Initializes all the required internals for the pipeline
/// \param batch_size
/// \param affinity
/// \param gpu_id
/// \param cpu_thread_count
/// \return
extern "C"  RocalContext  ROCAL_API_CALL rocalCreate(size_t batch_size, RocalProcessMode affinity, int gpu_id = 0, size_t cpu_thread_count = 1, size_t prefetch_queue_depth = 3, RocalTensorOutputType output_tensor_data_type = RocalTensorOutputType::ROCAL_FP32);
//extern "C"  RocalContext  ROCAL_API_CALL rocalCreate(size_t batch_size, RocalProcessMode affinity, int gpu_id = 0, size_t cpu_thread_count = 1);

///
/// \param context
/// \return
extern "C"  RocalStatus ROCAL_API_CALL rocalVerify(RocalContext context);

///
/// \param context
/// \return
extern "C"  RocalStatus  ROCAL_API_CALL rocalRun(RocalContext context);

///
/// \param rocal_context
/// \return
extern "C"  RocalStatus  ROCAL_API_CALL rocalRelease(RocalContext rocal_context);

#endif

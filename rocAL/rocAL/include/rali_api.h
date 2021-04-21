/*
MIT License
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

#ifndef RALI_H
#define RALI_H

#include "rali_api_types.h"
#include "rali_api_parameters.h"
#include "rali_api_data_loaders.h"
#include "rali_api_augmentation.h"
#include "rali_api_data_transfer.h"
#include "rali_api_meta_data.h"
#include "rali_api_info.h"

/// Creates the context for a new augmentation pipeline. Initializes all the required internals for the pipeline
/// \param batch_size
/// \param affinity
/// \param gpu_id
/// \param cpu_thread_count
/// \return
extern "C"  RaliContext  RALI_API_CALL raliCreate(size_t batch_size, RaliProcessMode affinity, int gpu_id = 0, size_t cpu_thread_count = 1);

///
/// \param context
/// \return
extern "C"  RaliStatus RALI_API_CALL raliVerify(RaliContext context);

///
/// \param context
/// \return
extern "C"  RaliStatus  RALI_API_CALL raliRun(RaliContext context);

///
/// \param rali_context
/// \return
extern "C"  RaliStatus  RALI_API_CALL raliRelease(RaliContext rali_context);

#endif

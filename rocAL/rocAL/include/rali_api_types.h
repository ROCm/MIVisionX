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

#ifndef MIVISIONX_RALI_API_TYPES_H
#define MIVISIONX_RALI_API_TYPES_H

#include <cstdlib>

#ifndef RALI_API_CALL
#if defined(_WIN32)
#define RALI_API_CALL __stdcall
#else
#define RALI_API_CALL
#endif
#endif

#include <half.hpp>
using half_float::half;

typedef void * RaliFloatParam;
typedef void * RaliIntParam;
typedef void * RaliContext;
typedef void * RaliImage;
typedef void * RaliMetaData;

struct TimingInfo
{
    long long unsigned load_time;
    long long unsigned decode_time;
    long long unsigned process_time;
    long long unsigned transfer_time;
};
enum RaliStatus
{
    RALI_OK = 0,
    RALI_CONTEXT_INVALID,
    RALI_RUNTIME_ERROR,
    RALI_UPDATE_PARAMETER_FAILED,
    RALI_INVALID_PARAMETER_TYPE
};


enum RaliImageColor
{
    RALI_COLOR_RGB24 = 0,
    RALI_COLOR_BGR24 = 1,
    RALI_COLOR_U8  = 2,
    RALI_COLOR_RGB_PLANAR = 3,
};

enum RaliProcessMode
{
    RALI_PROCESS_GPU = 0,
    RALI_PROCESS_CPU = 1
};

enum RaliFlipAxis
{
    RALI_FLIP_HORIZONTAL = 0,
    RALI_FLIP_VERTICAL = 1
};

enum RaliImageSizeEvaluationPolicy
{
    RALI_USE_MAX_SIZE = 0,
    RALI_USE_USER_GIVEN_SIZE = 1,
    RALI_USE_MOST_FREQUENT_SIZE = 2,
    RALI_USE_USER_GIVEN_SIZE_RESTRICTED = 3,    // use the given size only if the actual decoded size is greater than the given size
    RALI_USE_MAX_SIZE_RESTRICTED = 4,       // use max size if the actual decoded size is greater than max
};

enum RaliDecodeDevice
{
    RALI_HW_DECODE = 0,
    RALI_SW_DECODE = 1
};

enum RaliTensorLayout
{
    RALI_NHWC = 0,
    RALI_NCHW = 1
};

enum RaliTensorOutputType
{
    RALI_FP32 = 0,
    RALI_FP16 = 1
};

enum RaliDecoderType
{
    RALI_DECODER_TJPEG = 0,
    RALI_DECODER_OPENCV = 1,
    RALI_DECODER_VIDEO_FFMPEG_SW = 2,
    RALI_DECODER_VIDEO_FFMPEG_HW = 3
};


#endif //MIVISIONX_RALI_API_TYPES_H

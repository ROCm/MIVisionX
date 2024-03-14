/*
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

#pragma once

#include "video_decoder.h"

#ifdef ROCAL_VIDEO
class HardWareVideoDecoder : public VideoDecoder
{
public:
    //! Default constructor
    HardWareVideoDecoder();
    VideoDecoder::Status Initialize(const char *src_filename) override;
    VideoDecoder::Status Decode(unsigned char *output_buffer, unsigned seek_frame_number, size_t sequence_length, size_t stride, int out_width, int out_height, int out_stride, AVPixelFormat out_format) override;
    int seek_frame(AVRational avg_frame_rate, AVRational time_base, unsigned frame_number) override;
    void release() override;
    ~HardWareVideoDecoder() override;
private:
    const char *_src_filename = NULL;
    AVFormatContext *_fmt_ctx = NULL;
    AVCodecContext *_video_dec_ctx = NULL;
    AVCodec *_decoder = NULL;
    AVStream *_video_stream = NULL;
    int _video_stream_idx = -1;
    AVPixelFormat _dec_pix_fmt;
    int _codec_width, _codec_height;
    AVHWDeviceType *hwDeviceType;
    AVBufferRef *hw_device_ctx = NULL;
    int hw_decoder_init(AVCodecContext *ctx, const enum AVHWDeviceType type, AVBufferRef *hw_device_ctx);
};
#endif

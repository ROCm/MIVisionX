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

#pragma once

#include "video_decoder.h"

class FFMPEG_VIDEO_DECODER : public VideoDecoder {
public:
    //! Default constructor
    FFMPEG_VIDEO_DECODER();

    virtual VideoDecoder::Status Initialize(const char *src_filename) override;
    virtual int open_codec_context(int *stream_idx, AVCodecContext **dec_ctx, AVFormatContext *fmt_ctx) override;
    virtual VideoDecoder::Status Decode(unsigned char* output_buffer, unsigned seek_frame_number, size_t sequence_length) override;
    virtual int decode_packet(AVCodecContext *dec, const AVPacket *pkt) override;
    virtual void release() override;
    virtual int output_video_frame(AVFrame *frame) override;

    ~FFMPEG_VIDEO_DECODER() override;
private:
	AVFormatContext *_fmt_ctx = NULL;
	AVCodecContext *_video_dec_ctx = NULL;
	int _width, _height;
	enum AVPixelFormat _pix_fmt;
	AVStream *_video_stream = NULL;
	const char *_src_filename = NULL;
	const char *_video_dst_filename = NULL;

	uint8_t *_video_dst_data[4] = {NULL};
	int      _video_dst_linesize[4] = {0};
	int _video_dst_bufsize;

	int _video_stream_idx = -1;
	AVFrame *_frame = NULL, *_decframe = NULL;
	AVPacket _pkt;
	int _video_frame_count = 0;
	unsigned _video_count = 0;
	unsigned _nb_frames = 0;
	unsigned _skipped_frames = 0;
    bool _end_of_stream = false;
    int _got_pic = 0;
	const AVPixelFormat _dst_pix_fmt = AV_PIX_FMT_BGR24;
};

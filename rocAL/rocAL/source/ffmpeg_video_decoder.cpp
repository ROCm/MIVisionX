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

#include <stdio.h>
#include <commons.h>
#include "ffmpeg_video_decoder.h"

FFMPEG_VIDEO_DECODER::FFMPEG_VIDEO_DECODER(){};

int FFMPEG_VIDEO_DECODER::seek_frame(AVRational avg_frame_rate, AVRational time_base, unsigned frame_number)
{
    auto seek_time = av_rescale_q((int64_t)frame_number, av_inv_q(avg_frame_rate), AV_TIME_BASE_Q);
    int64_t select_frame_pts = av_rescale_q((int64_t)frame_number, av_inv_q(avg_frame_rate), time_base);
    int ret = av_seek_frame(_fmt_ctx, -1, seek_time, AVSEEK_FLAG_BACKWARD);
    if (ret < 0)
    {
        fprintf(stderr,"Error in seeking frame..Unable to seek the given frame in a video\n");
        return ret;
    }
    return select_frame_pts;
}

VideoDecoder::Status FFMPEG_VIDEO_DECODER::Decode(unsigned char *out_buffer, unsigned seek_frame_number, size_t sequence_length, size_t stride, int out_width, int out_height, int out_stride, AVPixelFormat out_pix_format)
{
    VideoDecoder::Status status = Status::OK;

    SwsContext *swsctx = nullptr;
    if ((out_width != _codec_width) || (out_height != _codec_height) || (out_pix_format != _dec_pix_fmt))
    {
      swsctx = sws_getCachedContext(nullptr, _codec_width, _codec_height, _dec_pix_fmt,
                                                out_width, out_height, out_pix_format, SWS_BILINEAR, nullptr, nullptr, nullptr);
      if (!swsctx)
      {
            fprintf(stderr, "Fail to get sws_getCachedContext\n");
            return Status::FAILED;
      }
    }

    AVFrame *dec_frame = av_frame_alloc();
    if (!dec_frame)
    {
        fprintf(stderr, "Could not allocate dec_frame\n");
        return Status::NO_MEMORY;
    }

    int select_frame_pts = seek_frame(_video_stream->avg_frame_rate, _video_stream->time_base, seek_frame_number);
    if (select_frame_pts < 0)
    {
        fprintf(stderr, "Error in seeking _frame..Unable to seek the given _frame in a video\n");
        return Status::FAILED;
    }

    AVPacket pkt;
    unsigned skipped_frames = 0;
    unsigned frame_count = 0;
    bool end_of_stream = false;
    int ret, got_pic = 0;
    uint8_t *dst_data[4] = {0};
    int dst_linesize[4] = {0};
    int image_size = out_height * out_stride * sizeof(unsigned char);

    do
    {
        if (!end_of_stream)
        {
            // read packet from input file
            ret = av_read_frame(_fmt_ctx, &pkt);
            if (ret < 0 && ret != AVERROR_EOF)
            {
                fprintf(stderr,"Fail to av_read_frame: ret=%d\n", ret);
                return Status::FAILED;
            }
            if (ret == 0 && pkt.stream_index != _video_stream_idx)
                continue;
            end_of_stream = (ret == AVERROR_EOF);
        }
        if (end_of_stream)
        {
            // null packet for bumping process
            av_init_packet(&pkt);
            pkt.data = nullptr;
            pkt.size = 0;
        }
        // decode video frame
        avcodec_decode_video2(_video_dec_ctx, dec_frame, &got_pic, &pkt);
        if ((dec_frame->pkt_pts < select_frame_pts) || !got_pic)
        {
            if (got_pic)
                ++skipped_frames;
            continue;
        }
        if (frame_count % stride == 0)
        {
            dst_data[0] = out_buffer;
            dst_linesize[0] = out_stride;
            if (swsctx)
              sws_scale(swsctx, dec_frame->data, dec_frame->linesize, 0, dec_frame->height, dst_data, dst_linesize);
            else
            {
                // copy from frame to out_buffer
                memcpy(out_buffer, dec_frame->data[0], dec_frame->linesize[0]*out_height);
            }
            out_buffer = out_buffer + image_size;
        }
        ++frame_count;
        if (frame_count == sequence_length * stride)
        {
            avcodec_flush_buffers(_video_dec_ctx);
            break;
        }
    } while (!end_of_stream || got_pic);

    av_free_packet(&pkt);
    av_frame_free(&dec_frame);
    return status;
}

// Initialize will open a new decoder and initialize the context
VideoDecoder::Status FFMPEG_VIDEO_DECODER::Initialize(const char *src_filename)
{
    VideoDecoder::Status status = Status::OK;
    int ret;
    AVDictionary *opts = NULL;

    /* open input file, and initialize the context required for decoding */
    _fmt_ctx = avformat_alloc_context();
    _src_filename = src_filename;
    if (avformat_open_input(&_fmt_ctx, src_filename, NULL, NULL) < 0)
    {
        fprintf(stderr, "Couldn't Open video file %s\n", src_filename);
        return Status::FAILED;
    }
    if (avformat_find_stream_info(_fmt_ctx, NULL) < 0)
    {
        fprintf(stderr, "av_find_stream_info error\n");
        return Status::FAILED;
    }
    ret = av_find_best_stream(_fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (ret < 0)
    {
        fprintf(stderr, "Could not find %s stream in input file '%s'\n",
                av_get_media_type_string(AVMEDIA_TYPE_VIDEO), src_filename);
        return Status::FAILED;
    }
    _video_stream_idx = ret;
    _video_stream = _fmt_ctx->streams[_video_stream_idx];
    _video_dec_ctx = _video_stream->codec;
    _dec_pix_fmt = _video_dec_ctx->pix_fmt;
    if (!_video_stream)
    {
        fprintf(stderr, "Could not find video stream in the input, aborting\n");
        release();
        return Status::FAILED;
    }
    /* find decoder for the stream */
    _decoder = avcodec_find_decoder(_video_stream->codecpar->codec_id);
    if (!_decoder)
    {
        fprintf(stderr, "Failed to find %s codec\n",
                av_get_media_type_string(AVMEDIA_TYPE_VIDEO));
        return Status::FAILED;
    }
    /* Allocate a codec context for the decoder */
    _video_dec_ctx = avcodec_alloc_context3(_decoder);
    if (!_video_dec_ctx)
    {
        fprintf(stderr, "Failed to allocate the %s codec context\n",
                av_get_media_type_string(AVMEDIA_TYPE_VIDEO));
        return Status::NO_MEMORY;
    }
    /* Copy codec parameters from input stream to output codec context */
    if ((ret = avcodec_parameters_to_context(_video_dec_ctx, _video_stream->codecpar)) < 0)
    {
        fprintf(stderr, "Failed to copy %s codec parameters to decoder context\n",
                av_get_media_type_string(AVMEDIA_TYPE_VIDEO));
        return Status::FAILED;
    }
    /* Init the decoders */
    if ((ret = avcodec_open2(_video_dec_ctx, _decoder, &opts)) < 0)
    {
        fprintf(stderr, "Failed to open %s codec\n",
                av_get_media_type_string(AVMEDIA_TYPE_VIDEO));
        return Status::FAILED;
    }
    _codec_width = _video_stream->codec->width;
    _codec_height = _video_stream->codec->height;
    return status;
}

void FFMPEG_VIDEO_DECODER::release()
{
    if (_video_dec_ctx) av_free(_video_dec_ctx);
    if (_fmt_ctx) av_free(_fmt_ctx);
}

FFMPEG_VIDEO_DECODER::~FFMPEG_VIDEO_DECODER()
{
    release();
}

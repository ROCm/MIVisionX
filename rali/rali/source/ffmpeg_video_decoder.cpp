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
    // std::cerr << "Seeking to frame " << frame_number << " timestamp " << seek_time << std::endl;    

    int ret = av_seek_frame(_fmt_ctx, -1, seek_time, AVSEEK_FLAG_BACKWARD);
    if (ret < 0) {
        std::cerr << "\n Error in seeking frame..Unable to seek the given frame in a video" << std::endl;
        return ret;
    }
    return select_frame_pts;
}

VideoDecoder::Status FFMPEG_VIDEO_DECODER::Decode(unsigned char *out_buffer, unsigned seek_frame_number, size_t sequence_length)
{

    VideoDecoder::Status status = Status::OK;
    int ret;

    /* Allocate a codec context for the decoder */
    _video_dec_ctx = avcodec_alloc_context3(_decoder);
    if (!_video_dec_ctx)
    {
        fprintf(stderr, "Failed to allocate the %s codec context\n",
                av_get_media_type_string(AVMEDIA_TYPE_VIDEO));
        return Status::NO_MEMORY;
    }

    /* Copy codec parameters from input stream to output codec context */
    if ((ret = avcodec_parameters_to_context(_video_dec_ctx, _video->codecpar)) < 0)
    {
        fprintf(stderr, "Failed to copy %s codec parameters to decoder context\n",
                av_get_media_type_string(AVMEDIA_TYPE_VIDEO));
        return Status::FAILED;
    }

    /* Init the decoders */
    if ((ret = avcodec_open2(_video_dec_ctx, _decoder, &_opts)) < 0)
    {
        fprintf(stderr, "Failed to open %s codec\n",
                av_get_media_type_string(AVMEDIA_TYPE_VIDEO));
        return Status::FAILED;
    }
    _video_stream = _fmt_ctx->streams[_video_stream_idx];

    // initialize sample scaler
    const int dst_width = _video_stream->codec->width;
    const int dst_height = _video_stream->codec->height;

    SwsContext *swsctx = sws_getCachedContext(nullptr, _video_stream->codec->width, _video_stream->codec->height, _video_stream->codec->pix_fmt,
                                              dst_width, dst_height, _dst_pix_fmt, SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!swsctx)
    {
        std::cerr << "fail to sws_getCachedContext";
        return Status::FAILED;
    }
    // std::cout << "output: " << dst_width << 'x' << dst_height << ',' << av_get_pix_fmt_name(_dst_pix_fmt) << std::endl;

    if (!_video_stream)
    {
        fprintf(stderr, "Could not find video stream in the input, aborting\n");
        release();
    }

    _frame = av_frame_alloc();
    if (!_frame)
    {
        fprintf(stderr, "Could not allocate _frame\n");
        status = Status::NO_MEMORY;
        release();
    }

    std::vector<uint8_t> framebuf(avpicture_get_size(_dst_pix_fmt, dst_width, dst_height));
    avpicture_fill(reinterpret_cast<AVPicture *>(_frame), framebuf.data(), _dst_pix_fmt, dst_width, dst_height);

    // decoding loop
    _decframe = av_frame_alloc();
    int fcount = 0;
    int select_frame_pts = seek_frame(_video_stream->avg_frame_rate, _video_stream->time_base, seek_frame_number);

    if (ret < 0)
    {
        std::cerr << "\n Error in seeking _frame..Unable to seek the given _frame in a video" << std::endl;
    }
    _skipped_frames = 0;
    do
    {
        if (!_end_of_stream)
        {
            // read packet from input file
            ret = av_read_frame(_fmt_ctx, &_pkt);
            if (ret < 0 && ret != AVERROR_EOF)
            {
                std::cerr << "fail to av_read_frame: ret=" << ret;
                return Status::FAILED;
            }
            if (ret == 0 && _pkt.stream_index != _video_stream_idx)
                goto next_packet;
            _end_of_stream = (ret == AVERROR_EOF);
        }
        if (_end_of_stream)
        {
            // null packet for bumping process
            av_init_packet(&_pkt);
            _pkt.data = nullptr;
            _pkt.size = 0;
        }
        // decode video _frame
        avcodec_decode_video2(_video_dec_ctx, _decframe, &_got_pic, &_pkt);

        if ((_decframe->pkt_pts < select_frame_pts) || !_got_pic)
        {
            if (_got_pic)
                ++_skipped_frames;
            goto next_packet;
        }

        _frame->data[0] = out_buffer;
        sws_scale(swsctx, _decframe->data, _decframe->linesize, 0, _decframe->height, _frame->data, _frame->linesize);

        ++_nb_frames;
        ++fcount;
        if (fcount == sequence_length)
        {
            av_free_packet(&_pkt);
            break;
        }
        out_buffer = out_buffer + (dst_height * dst_width * 3 * sizeof(unsigned char));
    next_packet:
        av_free_packet(&_pkt);
    } while (!_end_of_stream || _got_pic);
    return status;
}

VideoDecoder::Status FFMPEG_VIDEO_DECODER::Initialize(const char *src_filename)
{
    VideoDecoder::Status status = Status::OK;
    int ret;

    /* open input file, and allocate format context */
    _fmt_ctx = avformat_alloc_context();
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
                av_get_media_type_string(AVMEDIA_TYPE_VIDEO), _src_filename);
        return Status::FAILED;
    }

    _video_stream_idx = ret;
    _video = _fmt_ctx->streams[_video_stream_idx];

    /* find decoder for the stream */
    _decoder = avcodec_find_decoder(_video->codecpar->codec_id);
    if (!_decoder)
    {
        fprintf(stderr, "Failed to find %s codec\n",
                av_get_media_type_string(AVMEDIA_TYPE_VIDEO));
        return Status::FAILED;
    }
}

void FFMPEG_VIDEO_DECODER::release()
{
    avcodec_free_context(&_video_dec_ctx);
    avcodec_close(_video_stream->codec);
    avformat_close_input(&_fmt_ctx);
    av_frame_free(&_frame);
    av_frame_free(&_decframe);
}

FFMPEG_VIDEO_DECODER::~FFMPEG_VIDEO_DECODER()
{
    release();
}
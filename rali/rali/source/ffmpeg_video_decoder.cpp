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
#include "log.h"
#include "exception.h"
#include "ffmpeg_video_decoder.h"

FFMPEG_VIDEO_DECODER::FFMPEG_VIDEO_DECODER(){
};

/*VideoDecoder::Status FFMPEG_VIDEO_DECODER::output_video_frame(AVFrame *frame)
{
    if (frame->width != width || frame->height != height ||
        frame->format != pix_fmt) {
        fprintf(stderr, "Error: Width, height and pixel format have to be "
                "constant in a rawvideo file, but the width, height or "
                "pixel format of the input video changed:\n"
                "old: width = %d, height = %d, format = %s\n"
                "new: width = %d, height = %d, format = %s\n",
                width, height, av_get_pix_fmt_name(pix_fmt),
                frame->width, frame->height,
                av_get_pix_fmt_name(frame->format));
        return -1;
    }

    printf("video_frame n:%d coded_n:%d\n",
           video_frame_count++, frame->coded_picture_number);

    av_image_copy(video_dst_data, video_dst_linesize,
                  (const uint8_t **)(frame->data), frame->linesize,
                  pix_fmt, width, height);

    fwrite(video_dst_data[0], 1, video_dst_bufsize, video_dst_file);
    return 0;
}*/

VideoDecoder::Status FFMPEG_VIDEO_DECODER::decode_packet(AVCodecContext *dec, const AVPacket *pkt)
{
    int ret = 0;

// include fseek operation to point to the start of the file 

    // submit the packet to the decoder
    ret = avcodec_send_packet(dec, pkt);
    if (ret < 0) {
        THROW("Error submitting a packet for decoding " + TOSTR(av_err2str(ret)));
        return Status::FAILED;
    }

    // get all the available frames from the decoder
    while (ret >= 0) {
        ret = avcodec_receive_frame(dec, frame);
        if (ret < 0) {
            // those two return values are special and mean there is no output
            // frame available, but there were no errors during decoding
            if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN))
		return Status::OK;

            THROW("Error during decoding " + TOSTR(av_err2str(ret)));
            return Status::FAILED;
        }

        // write the frame data to output file
        if (dec->codec->type == AVMEDIA_TYPE_VIDEO)
            ret = output_video_frame(frame);
	av_frame_unref(frame);
        if (ret < 0)
           return Status::FAILED;
    }

    return Status::OK;
}

VideoDecoder::Status FFMPEG_VIDEO_DECODER::open_codec_context(int *stream_idx,
                              AVCodecContext **dec_ctx, AVFormatContext *fmt_ctx)
{
    int ret, stream_index;
    AVStream *st;
    AVCodec *dec = NULL;
    AVDictionary *opts = NULL;

    ret = av_find_best_stream(fmt_ctx, type, -1, -1, NULL, 0);
    if (ret < 0) {
        THROW("Could not find " + TOSTR(av_get_media_type_string(type)) +" stream in input file " + TOSTR(src_filename));
        return Status::FAILED;
    }
    else {
        stream_index = ret;
        st = fmt_ctx->streams[stream_index];

        /* find decoder for the stream */
        dec = avcodec_find_decoder(st->codecpar->codec_id);
        if (!dec) {
            THROW("Failed to find" + TOSTR(av_get_media_type_string(type)) + "codec\n");
	    return Status::FAILED;
        }

        /* Allocate a codec context for the decoder */
        *dec_ctx = avcodec_alloc_context3(dec);
        if (!*dec_ctx) {
            THROW("Failed to allocate the " + TOSTR(av_get_media_type_string(type)) + "codec context\n");
	    return Status::NO_MEMORY;
        }

        /* Copy codec parameters from input stream to output codec context */
        if ((ret = avcodec_parameters_to_context(*dec_ctx, st->codecpar)) < 0) {
            THROW("Failed to copy" + TOSTR(av_get_media_type_string(type)) + " codec parameters to decoder context\n");
            return Status::FAILED;
        }

        /* Init the decoders */
        if ((ret = avcodec_open2(*dec_ctx, dec, &opts)) < 0) {
            THROW("Failed to open " + TOSTR(av_get_media_type_string(type)) + " codec\n");
            return Status::FAILED;
        }
        *stream_idx = stream_index;
    }

    return Status::OK;
}

VideoDecoder::Status FFMPEG_VIDEO_DECODER::Decode(const char *src_filename, const char *video_dst_filename)
{
    VideoDecoder::Status status;
    int ret;
    /* open input file, and allocate format context */
    int err = avformat_open_input(&fmt_ctx, src_filename, NULL, NULL);
    if(err)
    {
	THROW("ERROR: avformat_open_input failed"+ TOSTR(err));
        return Status::FAILED;
    }

    /* retrieve stream information */
    err = avformat_find_stream_info(fmt_ctx, NULL);
    if(err)
    {
	THROW("ERROR: avformat_find_stream_info() failed " + TOSTR(err));
	return Status::FAILED;
    }

    if (open_codec_context(&video_stream_idx, &video_dec_ctx, fmt_ctx) >= 0) {
        video_stream = fmt_ctx->streams[video_stream_idx];

        video_dst_file = fopen(video_dst_filename, "wb");
        if (!video_dst_file) {
            THROW("Could not open destination file " + TOSTR(video_dst_filename));
            release();
        }

        /* allocate image where the decoded image will be put */
        width = video_dec_ctx->width;
        height = video_dec_ctx->height;
        pix_fmt = video_dec_ctx->pix_fmt;
        ret = av_image_alloc(video_dst_data, video_dst_linesize,
                             width, height, pix_fmt, 1);
        if (ret < 0) {
            THROW("Could not allocate raw video buffer\n");
            status = Status::NO_MEMORY;
	    release();
        }
        video_dst_bufsize = ret;
    }

    /* dump input information to stderr */
    av_dump_format(fmt_ctx, 0, src_filename, 0);

    if (!video_stream) {
        THROW("Could not find audio or video stream in the input, aborting\n");
        release();
    }

    frame = av_frame_alloc();
    if (!frame) {
        THROW("Could not allocate frame\n");
        status = Status::NO_MEMORY;
        release();
    }

    /* initialize packet, set data to NULL, let the demuxer fill it */
    av_init_packet(&pkt);
    pkt.data = NULL;
    pkt.size = 0;

    /* read frames from the file */
    while (av_read_frame(fmt_ctx, &pkt) >= 0) {
        // check if the packet belongs to a stream we are interested in, otherwise
        // skip it
        if (pkt.stream_index == video_stream_idx)
            ret = decode_packet(video_dec_ctx, &pkt);
        av_packet_unref(&pkt);
        if (ret < 0)
            break;
    }

    /* flush the decoders */
    if (video_dec_ctx)
        decode_packet(video_dec_ctx, NULL);

    std::cout << "Demuxing succeeded" << std::endl;

    if (video_stream) {
        std::cout << "Play the output video file with the command:";
        std::cout << "ffplay -f rawvideo -pix_fmt " << av_get_pix_fmt_name(pix_fmt) << "-video_size " << width << "x" 
		<< height << " " << video_dst_filename;
    }
    release();
    return status;
}

VideoDecoder::Status FFMPEG_VIDEO_DECODER::Initialize()
{
//Yet to Add
}

void FFMPEG_VIDEO_DECODER::release()
{
    avcodec_free_context(&video_dec_ctx);
    avcodec_free_context(&audio_dec_ctx);
    avformat_close_input(&fmt_ctx);
    if (video_dst_file)
        fclose(video_dst_file);
    av_frame_free(&frame);
    av_free(video_dst_data[0]);
}

FFMPEG_VIDEO_DECODER::~FFMPEG_VIDEO_DECODER() 
{
}

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

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

FFMPEG_VIDEO_DECODER::FFMPEG_VIDEO_DECODER(){
};

int FFMPEG_VIDEO_DECODER::output_video_frame(AVFrame *frame)
{
    if (frame->width != width || frame->height != height ||
        frame->format != pix_fmt) {
        /*fprintf(stderr, "Error: Width, height and pixel format have to be "
                "constant in a rawvideo file, but the width, height or "
                "pixel format of the input video changed:\n"
                "old: width = %d, height = %d, format = %s\n"
                "new: width = %d, height = %d, format = %s\n",
                width, height, av_get_pix_fmt_name(pix_fmt),
                frame->width, frame->height,
                av_get_pix_fmt_name(frame->format));*/
        return -1;
    }

    std::cout << "video_frame n:"<< video_frame_count++ << " coded_n:" << frame->coded_picture_number << std::endl;

    av_image_copy(video_dst_data, video_dst_linesize,
                  (const uint8_t **)(frame->data), frame->linesize,
                  pix_fmt, width, height);

    /*FILE *img_file;
    img_file = fopen ("img_out.yuv", "wb");
    fwrite(video_dst_data[0], 1, video_dst_bufsize, img_file);
    exit(0);*/
    fwrite(video_dst_data[0], 1, video_dst_bufsize, video_dst_file);
    return 0;
}

int FFMPEG_VIDEO_DECODER::decode_packet(AVCodecContext *dec, const AVPacket *pkt)
{
    int ret = 0;
    //include fseek operation to point to the start of the file

    // submit the packet to the decoder
    ret = avcodec_send_packet(dec, pkt);
    if (ret < 0)
    {
	fprintf(stderr, "Error submitting a packet for decoding\n");
	return ret;
    }

    // get all the available frames from the decoder
    while (ret >= 0)
    {
	ret = avcodec_receive_frame(dec, frame);
	if (ret < 0)
	{
	    // those two return values are special and mean there is no output
            // frame available, but there were no errors during decoding
            if (ret == AVERROR_EOF || ret == AVERROR(EAGAIN))
                return 0;

	    fprintf(stderr, "Error during decoding\n");
            return ret;
        }
       	// write the frame data to output file
        ret = output_video_frame(frame);
        av_frame_unref(frame);
      	if (ret < 0)
	    return ret;
    }
    return 0;
}

int FFMPEG_VIDEO_DECODER::open_codec_context(int *stream_idx,
                              AVCodecContext **dec_ctx, AVFormatContext *fmt_ctx)
{
    int ret, stream_index;
    AVStream *st;
    AVCodec *dec = NULL;
    AVDictionary *opts = NULL;

    ret = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (ret < 0) {
	fprintf(stderr, "Could not find %s stream in input file '%s'\n",
                av_get_media_type_string(AVMEDIA_TYPE_VIDEO), src_filename);	
        return ret;
    }
    else {
        stream_index = ret;
        st = fmt_ctx->streams[stream_index];

        /* find decoder for the stream */
        dec = avcodec_find_decoder(st->codecpar->codec_id);
        if (!dec) {
            fprintf(stderr, "Failed to find %s codec\n",
                    av_get_media_type_string(AVMEDIA_TYPE_VIDEO));
	    return AVERROR(EINVAL);
        }

        /* Allocate a codec context for the decoder */
        *dec_ctx = avcodec_alloc_context3(dec);
        if (!*dec_ctx) {
            fprintf(stderr, "Failed to allocate the %s codec context\n",
                    av_get_media_type_string(AVMEDIA_TYPE_VIDEO));
	    return AVERROR(ENOMEM);
        }

        /* Copy codec parameters from input stream to output codec context */
        if ((ret = avcodec_parameters_to_context(*dec_ctx, st->codecpar)) < 0) {
            fprintf(stderr, "Failed to copy %s codec parameters to decoder context\n",
                    av_get_media_type_string(AVMEDIA_TYPE_VIDEO));
	    return ret;
        }

        /* Init the decoders */
        if ((ret = avcodec_open2(*dec_ctx, dec, &opts)) < 0) {
            fprintf(stderr, "Failed to open %s codec\n",
                    av_get_media_type_string(AVMEDIA_TYPE_VIDEO));
	    return ret;
        }
        *stream_idx = stream_index;
    }
    return 0;
}

VideoDecoder::Status FFMPEG_VIDEO_DECODER::Decode(const char *src_filename, const char *video_dst_filename)
{
    VideoDecoder::Status status;
    int ret;
    /* open input file, and allocate format context */
    std::cerr << "\nSrc file name : " << src_filename << std::endl;
    int err = avformat_open_input(&fmt_ctx, src_filename, NULL, NULL);
    std::cerr << "\nRead source file : " << std::endl;
    if(err)
    {
        fprintf(stderr, "Could not open source file %s\n", src_filename);
	    return Status::FAILED;
    }
    
    std::cerr << "Before Retrieved stream info";
    /* retrieve stream information */
    err = avformat_find_stream_info(fmt_ctx, NULL);
    std::cerr << "Retrieved stream info";
    if(err)
    {
	fprintf(stderr, "Could not find stream information\n");	
	return Status::FAILED;
    }

    std::cerr << "\nBefore codec context: ";
    if (open_codec_context(&video_stream_idx, &video_dec_ctx, fmt_ctx) >= 0) {
        video_stream = fmt_ctx->streams[video_stream_idx];

        video_dst_file = fopen(video_dst_filename, "wb");
        if (!video_dst_file) {
            fprintf(stderr, "Could not open destination file %s\n", video_dst_filename);
	    release();
        }

        /* allocate image where the decoded image will be put */
        width = video_dec_ctx->width;
        height = video_dec_ctx->height;
        pix_fmt = video_dec_ctx->pix_fmt;
        std::cerr << "\n Width : " << width;
        std::cerr << "\n Height : " << height;
        ret = av_image_alloc(video_dst_data, video_dst_linesize,
                             width, height, pix_fmt, 1);
        if (ret < 0) {
	    fprintf(stderr, "Could not allocate raw video buffer\n");
            status = Status::NO_MEMORY;
	    release();
        }
        video_dst_bufsize = ret;
    }
    std::cerr << "\nAfter codec context";

    /* dump input information to stderr */
    av_dump_format(fmt_ctx, 0, src_filename, 0);

    if (!video_stream) {
	fprintf(stderr, "Could not find audio or video stream in the input, aborting\n");
        release();
    }

    frame = av_frame_alloc();
    if (!frame) {
        fprintf(stderr, "Could not allocate frame\n");        
        status = Status::NO_MEMORY;
        release();
    }

    /* initialize packet, set data to NULL, let the demuxer fill it */
    av_init_packet(&pkt);
    pkt.data = NULL;
    pkt.size = 0;

    std::cerr << "\nBefore reading a frame :" ;
    /* read frames from the file */
    while (av_read_frame(fmt_ctx, &pkt) >= 0) {

        		/*status = av_read_frame(inputMediaFormatContext[mediaIndex], &avpkt);
				if (status < 0) {
                    if ((status == AVERROR_EOF) && LoopDec[mediaIndex]) {
                        auto stream = inputMediaFormatContext[mediaIndex]->streams[videoStreamIndex[mediaIndex]];
                        avio_seek(inputMediaFormatContext[mediaIndex]->pb, 0, SEEK_SET);
                        avformat_seek_file(inputMediaFormatContext[mediaIndex], videoStreamIndex[mediaIndex], 0, 0, stream->duration, 0);
                */

        // check if the packet belongs to a stream we are interested in, otherwise
        // skip it
        if (pkt.stream_index == video_stream_idx) 
            ret = decode_packet(video_dec_ctx, &pkt);
        std::cerr << "\nDecoding is done : " ;
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

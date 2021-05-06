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

FFMPEG_VIDEO_DECODER::FFMPEG_VIDEO_DECODER(){};

int FFMPEG_VIDEO_DECODER::output_video_frame(AVFrame *frame)
{
    /*    if (frame->width != width || frame->height != height ||
        frame->format != pix_fmt)
    {
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

    std::cout << "video_frame n:" << video_frame_count++ << " coded_n:" << frame->coded_picture_number << std::endl;

    av_image_copy(video_dst_data, video_dst_linesize,
                  (const uint8_t **)(frame->data), frame->linesize,
                  pix_fmt, width, height);

    FILE *img_file;
    img_file = fopen("img_out.yuv", "wb");

    int y_size = frame->width * frame->height;
    fwrite(video_dst_data[0], 1, y_size, img_file);     //Y
    fwrite(video_dst_data[1], 1, y_size / 4, img_file); //U
    fwrite(video_dst_data[2], 1, y_size / 4, img_file); //V

    //fwrite(video_dst_data[0], 1, height*frame->linesize[0], img_file);
    //exit(0);
    fwrite(video_dst_data[0], 1, video_dst_bufsize, video_dst_file);*/
    return 0;
}

int FFMPEG_VIDEO_DECODER::decode_packet(AVCodecContext *dec, const AVPacket *pkt)
{
    /*    int ret = 0;
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
    }*/
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
    if (ret < 0)
    {
        fprintf(stderr, "Could not find %s stream in input file '%s'\n",
                av_get_media_type_string(AVMEDIA_TYPE_VIDEO), src_filename);
        return ret;
    }
    else
    {
        stream_index = ret;
        st = fmt_ctx->streams[stream_index];

        /* find decoder for the stream */
        dec = avcodec_find_decoder(st->codecpar->codec_id);
        if (!dec)
        {
            fprintf(stderr, "Failed to find %s codec\n",
                    av_get_media_type_string(AVMEDIA_TYPE_VIDEO));
            return AVERROR(EINVAL);
        }

        /* Allocate a codec context for the decoder */
        *dec_ctx = avcodec_alloc_context3(dec);
        if (!*dec_ctx)
        {
            fprintf(stderr, "Failed to allocate the %s codec context\n",
                    av_get_media_type_string(AVMEDIA_TYPE_VIDEO));
            return AVERROR(ENOMEM);
        }

        /* Copy codec parameters from input stream to output codec context */
        if ((ret = avcodec_parameters_to_context(*dec_ctx, st->codecpar)) < 0)
        {
            fprintf(stderr, "Failed to copy %s codec parameters to decoder context\n",
                    av_get_media_type_string(AVMEDIA_TYPE_VIDEO));
            return ret;
        }

        /* Init the decoders */
        if ((ret = avcodec_open2(*dec_ctx, dec, &opts)) < 0)
        {
            fprintf(stderr, "Failed to open %s codec\n",
                    av_get_media_type_string(AVMEDIA_TYPE_VIDEO));
            return ret;
        }
        *stream_idx = stream_index;
    }
    return 0;
}

/* int64_t FFMPEG_VIDEO_DECODER::seek_frame(AVFormatContext fmt_ctx, AVRational avg_frame_rate, AVRational time_base, AV, unsigned frame_number)
{
    auto seek_time = av_rescale_q((int64_t)frame_number, av_inv_q(avg_frame_rate), AV_TIME_BASE_Q);
    int64_t select_frame_pts = av_rescale_q((int64_t)frame_number, av_inv_q(avg_frame_rate), time_base);
    // std::cerr << "Seeking to frame " << frame_number << " timestamp " << seek_time << std::endl;    

    int ret = av_seek_frame(fmt_ctx, -1, seek_time, AVSEEK_FLAG_BACKWARD);
    if (ret < 0) {
        std::cerr << "\n Error in seeking frame..Unable to seek the given frame in a video" << std::endl;
    }
}
*/

VideoDecoder::Status FFMPEG_VIDEO_DECODER::Decode(unsigned char *out_buffer, const char *src_filename, unsigned seek_frame_number, size_t sequence_length)
{

    VideoDecoder::Status status = Status::OK;
    int ret;
    /* open input file, and allocate format context */
    // std::cerr << "\nThe source file name in Decode: "<<src_filename<<"\t";
    // std::cerr << " start : " << seek_frame_number << "\n";
    fmt_ctx = avformat_alloc_context();
    if (avformat_open_input(&fmt_ctx, src_filename, NULL, NULL) < 0)
    {
        //if(av_open_input_file(&pFormatCtx, videofile, NULL, 0, NULL) < 0){
        fprintf(stderr, "Couldn't Open video file %s\n", src_filename);
        return Status::FAILED;
    }

    if (avformat_find_stream_info(fmt_ctx, NULL) < 0)
    {
        //	av_close_input_file(pFormatCtx);
        fprintf(stderr, "av_find_stream_info error\n");
        return Status::FAILED; // Couldn't open file
    }

    if (open_codec_context(&video_stream_idx, &video_dec_ctx, fmt_ctx) >= 0)
    {
        video_stream = fmt_ctx->streams[video_stream_idx];

        // print input video stream informataion
        // std::cout
        //     << "source file: " << src_filename << "\n"
        //     << "format: " << fmt_ctx->iformat->name << "\n"
        //     //<< "vcodec: " << video_dec_ctx->name << "\n"
        //     << "size:   " << video_stream->codec->width << 'x' << video_stream->codec->height << "\n"
        //     << "fps:    " << av_q2d(video_stream->codec->framerate) << " [fps]\n"
        //     << "length: " << av_rescale_q(video_stream->duration, video_stream->time_base, {1, 1000}) / 1000. << " [sec]\n"
        //     << "pixfmt: " << av_get_pix_fmt_name(video_stream->codec->pix_fmt) << "\n"
        //     << "frame:  " << video_stream->nb_frames << "\n"
        //     << std::flush;
    }

    // initialize sample scaler
    const int dst_width = video_stream->codec->width;
    const int dst_height = video_stream->codec->height;
    SwsContext *swsctx = sws_getCachedContext(nullptr, video_stream->codec->width, video_stream->codec->height, video_stream->codec->pix_fmt,
        dst_width, dst_height, dst_pix_fmt, SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!swsctx)
    {
        std::cerr << "fail to sws_getCachedContext";
        return Status::FAILED;
    }
    // std::cout << "output: " << dst_width << 'x' << dst_height << ',' << av_get_pix_fmt_name(dst_pix_fmt) << std::endl;

    if (!video_stream)
    {
        fprintf(stderr, "Could not find video stream in the input, aborting\n");
        release();
    }

    frame = av_frame_alloc(); // check the format, height & width & linesize
    if (!frame)
    {
        fprintf(stderr, "Could not allocate frame\n");
        status = Status::NO_MEMORY;
        release();
    }

    std::vector<uint8_t> framebuf(avpicture_get_size(dst_pix_fmt, dst_width, dst_height));
    avpicture_fill(reinterpret_cast<AVPicture *>(frame), framebuf.data(), dst_pix_fmt, dst_width, dst_height);

    // decoding loop
    decframe = av_frame_alloc();
    int fcount = 0;

    
    auto seek_time = av_rescale_q((int64_t)seek_frame_number, av_inv_q(video_stream->avg_frame_rate), AV_TIME_BASE_Q);
    int64_t select_frame_pts = av_rescale_q((int64_t)seek_frame_number, av_inv_q(video_stream->avg_frame_rate), video_stream->time_base);
    // std::cerr << "Seeking to frame " << seek_frame_number << " timestamp " << seek_time << std::endl;    

    ret = av_seek_frame(fmt_ctx, -1, seek_time, AVSEEK_FLAG_BACKWARD);
    if (ret < 0) {
        std::cerr << "\n Error in seeking frame..Unable to seek the given frame in a video" << std::endl;
    }
    
//    int64_t select_frame_pts = seek_frame(fmt_ctx, video_stream->avg_frame_rate, video_stream->time_base, seek_frame_number);

// has to be changed to decode only the part which we need seek operations ll come here
    skipped_frames = 0;
    do
    {
        if (!end_of_stream)
        {
            // read packet from input file
            ret = av_read_frame(fmt_ctx, &pkt);
            if (ret < 0 && ret != AVERROR_EOF)
            {
                std::cerr << "fail to av_read_frame: ret=" << ret;
                return Status::FAILED;
            }
            if (ret == 0 && pkt.stream_index != video_stream_idx)
                goto next_packet;
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
        avcodec_decode_video2(video_dec_ctx, decframe, &got_pic, &pkt);

        if ((decframe->pkt_pts < select_frame_pts) || !got_pic)
        {
            if (got_pic)
                ++skipped_frames;
            goto next_packet;
        }

        //convert frame to OpenCV matrix
        frame->data[0] = out_buffer;
        sws_scale(swsctx, decframe->data, decframe->linesize, 0, decframe->height, frame->data, frame->linesize);

        /*  OpenCV image writer module
        std::cerr << "\n Frame line size :" << frame->linesize[0];
        // memcpy(out_buffer, frame->data, dst_height * frame->linesize[0]);
        {
            // cv::Mat image_1(dst_height, dst_width, CV_8UC3, video_dst_data, video_dst_linesize[0]);
            cv::Mat image(dst_height, dst_width, CV_8UC3, frame->data[0], frame->linesize[0]);
            //cv::imshow("press ESC to exit", image);
            std::string filename = "output_swssmallout_image";
            filename.append(std::to_string(nb_frames));
            filename.append(".png");
            cv::imwrite(filename.c_str(), image);
            // cv::imwrite("frame.png", image_1);
            // if (cv::waitKey(1) == 0x1b)
                // break;
        }
        std::cout << nb_frames << '\r' << std::flush; // dump progress
        */

        ++nb_frames;
        ++fcount;
        if( fcount == sequence_length)
        {
            av_free_packet(&pkt);
            break;
        }
        out_buffer = out_buffer + (dst_height * dst_width * 3 * sizeof(unsigned char));
    next_packet:
        av_free_packet(&pkt);
    } while (!end_of_stream || got_pic);
    // std::cout << nb_frames << " frames decoded" << std::endl;

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
    avcodec_close(video_stream->codec);
    avformat_close_input(&fmt_ctx);
    av_frame_free(&frame);
    av_frame_free(&decframe);
}

FFMPEG_VIDEO_DECODER::~FFMPEG_VIDEO_DECODER()
{
}
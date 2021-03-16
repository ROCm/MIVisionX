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
#include <libavutil/imgutils.h>
#include <libavutil/samplefmt.h>
#include <libavutil/timestamp.h>
#include <libavformat/avformat.h>

class FFMPEG_VIDEO_DECODER : public VideoDecoder {
public:
    //! Default constructor
    FFMPEG_VIDEO_DECODER();

    //! Decodes the header of the Jpeg compressed data and returns basic info about the compressed image
    /*!
     \param input_buffer  User provided buffer containig the encoded image
     \param input_size Size of the compressed data provided in the input_buffer
     \param width pointer to the user's buffer to write the width of the compressed image to 
     \param height pointer to the user's buffer to write the height of the compressed image to 
     \param color_comps pointer to the user's buffer to write the number of color components of the compressed image to 
    */
    //Status decode_info(unsigned char* input_buffer, size_t input_size, int* width, int* height, int* color_comps) override;
    
    //! Decodes the actual image data
    /*! 
      \param input_buffer  User provided buffer containig the encoded image
      \param output_buffer User provided buffer used to write the decoded image into
      \param input_size Size of the compressed data provided in the input_buffer
      \param max_decoded_width The maximum width user wants the decoded image to be. Image will be downscaled if bigger.
      \param max_decoded_height The maximum height user wants the decoded image to be. Image will be downscaled if bigger.
      \param original_image_width The actual width of the compressed image. decoded width will be equal to this if this is smaller than max_decoded_width
      \param original_image_height The actual height of the compressed image. decoded height will be equal to this if this is smaller than max_decoded_height
    */
    /*Decoder::Status decode(unsigned char *input_buffer, size_t input_size, unsigned char *output_buffer,
                           size_t max_decoded_width, size_t max_decoded_height,
                           size_t original_image_width, size_t original_image_height,
                           size_t &actual_decoded_width, size_t &actual_decoded_height,
                           Decoder::ColorFormat desired_decoded_color_format, DecoderConfig config, bool keep_original_size=false) override;*/
    ~FFMPEG_VIDEO_DECODER() override;
private:
	AVFormatContext *fmt_ctx = NULL;
	AVCodecContext *video_dec_ctx = NULL, *audio_dec_ctx;
	int width, height;
	enum AVPixelFormat pix_fmt;
	AVStream *video_stream = NULL, *audio_stream = NULL;
	const char *src_filename = NULL;
	const char *video_dst_filename = NULL;
	const char *audio_dst_filename = NULL;
	FILE *video_dst_file = NULL;
	FILE *audio_dst_file = NULL;

	uint8_t *video_dst_data[4] = {NULL};
	int      video_dst_linesize[4];
	int video_dst_bufsize;

	int video_stream_idx = -1, audio_stream_idx = -1;
	AVFrame *frame = NULL;
	AVPacket pkt;
	int video_frame_count = 0;
	int audio_frame_count = 0;
};

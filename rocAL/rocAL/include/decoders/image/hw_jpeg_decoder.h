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

#include "decoder.h"
#ifdef ROCAL_VIDEO
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
#include <libavutil/hwcontext.h>
#include <libavformat/avio.h>
#include <libavutil/file.h>
}


class HWJpegDecoder : public Decoder {
public:
    //! Default constructor
    HWJpegDecoder() {};
    //! Decodes the header of the Jpeg compressed data and returns basic info about the compressed image
    /*!
     \param input_buffer  User provided buffer containig the encoded image
     \param input_size Size of the compressed data provided in the input_buffer
     \param width pointer to the user's buffer to write the width of the compressed image to
     \param height pointer to the user's buffer to write the height of the compressed image to
     \param color_comps pointer to the user's buffer to write the number of color components of the compressed image to
    */
    Status decode_info(unsigned char* input_buffer, size_t input_size, int* width, int* height, int* color_comps) override;

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
    Decoder::Status decode(unsigned char *input_buffer, size_t input_size, unsigned char *output_buffer,
                           size_t max_decoded_width, size_t max_decoded_height,
                           size_t original_image_width, size_t original_image_height,
                           size_t &actual_decoded_width, size_t &actual_decoded_height,
                           Decoder::ColorFormat desired_decoded_color_format, DecoderConfig config, bool keep_original_size=false) override;

    ~HWJpegDecoder() override;
    void initialize(int device_id=0);
    bool is_partial_decoder() override { return _is_partial_decoder; }
    void set_bbox_coords(std::vector <float> bbox_coord) override { _bbox_coord = bbox_coord;}
    void set_crop_window(CropWindow &crop_window) override { _crop_window = crop_window;}
    std::vector <float> get_bbox_coords() override { return _bbox_coord;}

private:
    void release();
    const char *_src_filename = NULL;
    AVHWDeviceType _hw_type = AV_HWDEVICE_TYPE_NONE;
    AVBufferRef *_hw_device_ctx = NULL;
    AVIOContext *_io_ctx = NULL;
    AVFormatContext *_fmt_ctx = NULL;
    AVCodecContext *_video_dec_ctx = NULL;
    AVCodec *_decoder = NULL;
    AVStream *_video_stream = NULL;
    int _video_stream_idx = -1;
    AVPixelFormat _dec_pix_fmt;
    size_t _codec_width, _codec_height;

    bool _is_partial_decoder = false;
    std::vector <float> _bbox_coord;
    CropWindow _crop_window;
};

#endif

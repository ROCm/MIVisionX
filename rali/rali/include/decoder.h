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

#include <cstddef>
#include <iostream>
#include <vector>
#include "parameter_factory.h"
enum class DecoderType
{
    TURBO_JPEG = 0,//!< Can only decode
    FUSED_TURBO_JPEG = 1, //!< FOR PARTIAL DECODING
    OPENCV_DEC = 2, //!< for back_up decoding
    SKIP_DECODE  = 3, //!< For skipping decoding in case of uncompressed data from reader
    OVX_FFMPEG,//!< Uses FFMPEG to decode video streams, can decode up to 4 video streams simultaneously
};



class DecoderConfig
{
public:
    DecoderConfig() {}
    explicit DecoderConfig(DecoderType type):_type(type){}
    virtual DecoderType type() {return _type; };
    DecoderType _type = DecoderType::TURBO_JPEG;
    std::vector<Parameter<float>*> _crop_param;
    void set_crop_param(std::vector<Parameter<float>*> crop_param) { _crop_param = std::move(crop_param); };
    std::vector<float> get_crop_param(){
        std::vector<float> crop_mul(4);
        _crop_param[0]->renew();
        crop_mul[0] = _crop_param[0]->get();
        _crop_param[1]->renew();
        crop_mul[1] = _crop_param[1]->get();
        _crop_param[2]->renew();
        crop_mul[2] = _crop_param[2]->get();
        _crop_param[3]->renew();
        crop_mul[3] = _crop_param[3]->get();
        return crop_mul;
    };
};


class Decoder
{
public:

    enum class Status {
        OK = 0,
        HEADER_DECODE_FAILED,
        CONTENT_DECODE_FAILED,
        UNSUPPORTED
    };

    enum class ColorFormat {
        GRAY = 0,
        RGB, 
        BGR
    };
    //! Decodes the header of the Jpeg compressed data and returns basic info about the compressed image
    /*!
     \param input_buffer  User provided buffer containig the encoded image
     \param input_size Size of the compressed data provided in the input_buffer
     \param width pointer to the user's buffer to write the width of the compressed image to 
     \param height pointer to the user's buffer to write the height of the compressed image to 
     \param color_comps pointer to the user's buffer to write the number of color components of the compressed image to 
    */
    virtual Status decode_info(unsigned char* input_buffer,
                                size_t input_size,
                                int* width, 
                                int* height, 
                                int* color_comps) = 0;
    
    // TODO: Extend the decode API if needed, color format and order can be passed to the function
    //! Decodes the actual image data
    /*! 
      \param input_buffer  User provided buffer containig the encoded image
      \param output_buffer User provided buffer used to write the decoded image into
      \param input_size Size of the compressed data provided in the input_buffer
      \param max_decoded_width The maximum width user wants the decoded image to be
      \param max_decoded_height The maximum height user wants the decoded image to be.

    */
    virtual Decoder::Status decode(unsigned char *input_buffer, size_t input_size, unsigned char *output_buffer,
                                   size_t max_decoded_width, size_t max_decoded_height,
                                   size_t original_image_width, size_t original_image_height,
                                   size_t &actual_decoded_width, size_t &actual_decoded_height,
                                   Decoder::ColorFormat desired_decoded_color_format, DecoderConfig decoder_config, bool keep_original) = 0;

    virtual ~Decoder() = default;
    virtual bool is_partial_decoder() = 0;
    virtual void set_bbox_coords(std::vector <float> bbox_coords) = 0;
    virtual std::vector <float> get_bbox_coords() = 0;
};

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
#include "open_cv_decoder.h"

#if ENABLE_OPENCV
int handleError( int status, const char* func_name,
            const char* err_msg, const char* file_name,
            int line, void* userdata )
{
    //TODO: add proper error handling here
    return 0;   
}

CVDecoder::CVDecoder() {
    cv::redirectError(handleError);
}

Decoder::Status CVDecoder::decode_info(unsigned char* input_buffer, size_t input_size, int* width, int* height, int* color_comps) {
    //TODO: OpenCV seems not able to decode header separately, remove the imdecode from this call if possible and replace it with a proper function for decoding the header only
    m_mat_orig = cv::imdecode(cv::Mat(1, input_size, CV_8UC1, input_buffer), cv::IMREAD_UNCHANGED);
    if(m_mat_orig.rows == 0 || m_mat_orig.cols == 0) {
        WRN("CVDecoder::Jpeg header decode failed ");
        return Status::HEADER_DECODE_FAILED;
    }
    *width = m_mat_orig.cols;
    *height = m_mat_orig.rows;
    *color_comps = 0;       // not known
    return Status::OK;
}

Decoder::Status CVDecoder::decode(unsigned char *input_buffer, size_t input_size, unsigned char *output_buffer,
                           size_t max_decoded_width, size_t max_decoded_height,
                           size_t original_image_width, size_t original_image_height,
                           size_t &actual_decoded_width, size_t &actual_decoded_height,
                           Decoder::ColorFormat desired_decoded_color_format, DecoderConfig config, bool keep_original_size) {

    m_mat_orig = cv::imdecode(cv::Mat(1, input_size, CV_8UC1, input_buffer), CV_LOAD_IMAGE_COLOR);
    if(m_mat_orig.rows == 0 || m_mat_orig.cols == 0) {
        return Status::CONTENT_DECODE_FAILED;
    }
    cv::resize(m_mat_orig, m_mat_scaled, cv::Size(max_decoded_width, max_decoded_height), cv::INTER_LINEAR);
    if(m_mat_scaled.rows == 0 || m_mat_scaled.cols == 0) {
        actual_decoded_width = m_mat_orig.cols;
        actual_decoded_height = m_mat_orig.rows;
    }else
    {
        actual_decoded_width = m_mat_scaled.cols;
        actual_decoded_height = m_mat_scaled.rows;
    }
    return Decoder::Status::OK;
}

CVDecoder::~CVDecoder() {
    m_mat_scaled.release();
    m_mat_orig.release();
}
#endif

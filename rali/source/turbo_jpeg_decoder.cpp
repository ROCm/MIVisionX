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

#include "turbo_jpeg_decoder.h"
#include <commons.h>
#include <stdio.h>

TJDecoder::TJDecoder() {
  m_jpegDecompressor = tjInitDecompress();

#if 0
    int num_avail_scalings = 0;
    auto scaling_factors = tjGetScalingFactors	(&num_avail_scalings);
    for(int i = 0; i < num_avail_scalings; i++) {
        if(scaling_factors[i].num < scaling_factors[i].denom) {

            printf("%d / %d  - ",scaling_factors[i].num, scaling_factors[i].denom );
        }
    }
#endif
};

Decoder::Status TJDecoder::decode_info(unsigned char *input_buffer,
                                       size_t input_size, int *width,
                                       int *height, int *color_comps) {
  // TODO : Use the most recent TurboJpeg API tjDecompressHeader3 which returns
  // the color components
  if (tjDecompressHeader2(m_jpegDecompressor, input_buffer, input_size, width,
                          height, color_comps) != 0) {
    WRN("Jpeg header decode failed " + STR(tjGetErrorStr2(m_jpegDecompressor)))
    return Status::HEADER_DECODE_FAILED;
  }
  return Status::OK;
}

Decoder::Status
TJDecoder::decode(unsigned char *input_buffer, size_t input_size,
                  unsigned char *output_buffer, size_t max_decoded_width,
                  size_t max_decoded_height, size_t original_image_width,
                  size_t original_image_height, size_t &actual_decoded_width,
                  size_t &actual_decoded_height,
                  Decoder::ColorFormat desired_decoded_color_format) {
  int tjpf = TJPF_RGB;
  int planes = 1;
  switch (desired_decoded_color_format) {
  case Decoder::ColorFormat::GRAY:
    tjpf = TJPF_GRAY;
    planes = 1;
    break;
  case Decoder::ColorFormat::RGB:
    tjpf = TJPF_RGB;
    planes = 3;
    break;
  case Decoder::ColorFormat::BGR:
    tjpf = TJPF_BGR;
    planes = 3;
    break;
  };
  actual_decoded_width = max_decoded_width;
  actual_decoded_height = max_decoded_height;

  // TODO : Turbo Jpeg supports multiple color packing and color formats, add
  // more as an option to the API TJPF_RGB, TJPF_BGR, TJPF_RGBX, TJPF_BGRX,
  // TJPF_RGBA, TJPF_GRAY, TJPF_CMYK , ...
  if (tjDecompress2(m_jpegDecompressor, input_buffer, input_size, output_buffer,
                    max_decoded_width, max_decoded_width * planes,
                    max_decoded_height, tjpf, TJFLAG_FASTDCT) != 0) {
    WRN("Jpeg image decode failed " + STR(tjGetErrorStr2(m_jpegDecompressor)))
    return Status::CONTENT_DECODE_FAILED;
  }
  // Find the decoded image size using the predefined scaling factors in the
  // turbo jpeg decoder
  uint scaledw = max_decoded_width, scaledh = max_decoded_height;
  for (auto scaling_factor : SCALING_FACTORS) {
    scaledw = TJSCALED(original_image_width, scaling_factor);
    scaledh = TJSCALED(original_image_height, scaling_factor);
    if (scaledw <= max_decoded_width && scaledh <= max_decoded_height) {
      break;
    }
  }
  actual_decoded_width = scaledw;
  actual_decoded_height = scaledh;
  return Status::OK;
}

TJDecoder::~TJDecoder() { tjDestroy(m_jpegDecompressor); }
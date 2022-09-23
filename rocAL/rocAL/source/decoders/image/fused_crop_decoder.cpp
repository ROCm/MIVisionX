/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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
#include <string.h>
#include "fused_crop_decoder.h"

FusedCropTJDecoder::FusedCropTJDecoder()
{
    m_jpegDecompressor = tjInitDecompress();

#if 0
    int num_avail_scalings = 0;
    auto scaling_factors = tjGetScalingFactors (&num_avail_scalings);
    for(int i = 0; i < num_avail_scalings; i++) {
        if(scaling_factors[i].num < scaling_factors[i].denom) {

            printf("%d / %d  - ",scaling_factors[i].num, scaling_factors[i].denom );
        }
    }
#endif
    //generate BatchSize of RNG's- Using a random seed
    int64_t seed = getseed();
    generate_rngs(seed, 256);
};

Decoder::Status FusedCropTJDecoder::decode_info(unsigned char* input_buffer, size_t input_size, int* width, int* height, int* color_comps)
{
    //TODO : Use the most recent TurboJpeg API tjDecompressHeader3 which returns the color components
    if(tjDecompressHeader2(m_jpegDecompressor,
                            input_buffer,
                            input_size,
                            width,
                            height,
                            color_comps) != 0)
    {
        WRN("Jpeg header decode failed " + STR(tjGetErrorStr2(m_jpegDecompressor)))
        return Status::HEADER_DECODE_FAILED;
    }
    return Status::OK;
}

Decoder::Status FusedCropTJDecoder::decode(unsigned char *input_buffer, size_t input_size, unsigned char *output_buffer,
                                  size_t max_decoded_width, size_t max_decoded_height,
                                  size_t original_image_width, size_t original_image_height,
                                  size_t &actual_decoded_width, size_t &actual_decoded_height,
                                  Decoder::ColorFormat desired_decoded_color_format, DecoderConfig decoder_config, bool keep_original_size)
{
    int tjpf = TJPF_RGB;
    int planes = 1;
    switch (desired_decoded_color_format) {
        case Decoder::ColorFormat::GRAY:
            tjpf = TJPF_GRAY;
            planes= 1;
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
    // You need get the output of random bbox crop
    // check the vector size for bounding box. If its more than zero go for random bbox crop
    // else go to random crop
    unsigned int crop_width, crop_height, x1, y1, x1_diff, crop_width_diff;
    if(_bbox_coord.size() != 0)
    {
        // Random bbox crop returns normalized crop cordinates
        // hence bringing it back to absolute cordinates
        x1 = std::lround(_bbox_coord[0] * original_image_width);
        y1 = std::lround(_bbox_coord[1] * original_image_height);
        crop_width = std::lround((_bbox_coord[2]) * original_image_width);
        crop_height = std::lround((_bbox_coord[3]) * original_image_height);
    }
    else
    {
        double ASPECT_RATIO_RANGE[2] = {decoder_config.get_random_aspect_ratio()[0], decoder_config.get_random_aspect_ratio()[1]};
        double AREA_RANGE[2] = {decoder_config.get_random_area()[0], decoder_config.get_random_aspect_ratio()[1]};
        auto is_valid_crop = [](uint h, uint w, uint height, uint width)
        {
            return (0 < h && h <= height && 0 < w && w <= width);
        };
  float max_wh_ratio = ASPECT_RATIO_RANGE[1];
  float max_hw_ratio = 1 / ASPECT_RATIO_RANGE[0];
  float min_area = original_image_width * original_image_height * AREA_RANGE[0];
  int maxW = std::max<int>(1, original_image_height * max_wh_ratio);
  int maxH = std::max<int>(1, original_image_width * max_hw_ratio);
  int num_of_attempts = decoder_config.get_num_attemps();
  int num_attempts_left = num_of_attempts;
  // detect two impossible cases early
  if (original_image_height * maxW < min_area) {  // image too wide
    crop_width = original_image_height;
    crop_height = maxW;
    x1 = std::uniform_int_distribution<int>(0, original_image_width - crop_width)(_rand_gen);
    y1 = std::uniform_int_distribution<int>(0, original_image_height - crop_height)(_rand_gen);
  } else if (original_image_width * maxH < min_area) {  // image too tall
    crop_width = maxH;
    crop_height = original_image_width;
    x1 = std::uniform_int_distribution<int>(0, original_image_width - crop_width)(_rand_gen);
    y1 = std::uniform_int_distribution<int>(0, original_image_height - crop_height)(_rand_gen);
  }
        else
        {

        for(; num_attempts_left > 0; num_attempts_left--)
        {
            std::uniform_real_distribution<double> area_dis(AREA_RANGE[0], AREA_RANGE[1]);
            std::uniform_real_distribution<double> log_ratio_dist(std::log(ASPECT_RATIO_RANGE[0]), std::log(ASPECT_RATIO_RANGE[1]));
            double scale = area_dis(_rand_gen);
            double target_area  = scale * original_image_width * original_image_height;
            double aspect_ratio = std::exp(log_ratio_dist(_rand_gen));
            crop_width  = static_cast<size_t>(std::round(std::sqrt(target_area * aspect_ratio)));
            crop_height = static_cast<size_t>(std::round(std::sqrt(target_area * (1 / aspect_ratio))));
            if(is_valid_crop(crop_height, crop_width, original_image_height, original_image_width))
            {
                x1 = std::uniform_int_distribution<int>(0, original_image_width - crop_width)(_rand_gen);
                y1 = std::uniform_int_distribution<int>(0, original_image_height - crop_height)(_rand_gen);
                break ;
            }
        }}
        // Fallback on Central Crop
        if(num_attempts_left <= 0)
        {
            double in_ratio;
            float max_area = AREA_RANGE[1] * original_image_height * original_image_width;

            in_ratio = static_cast<double>(original_image_width) / original_image_height;
            if(in_ratio < ASPECT_RATIO_RANGE[0])
            {
                crop_width =  original_image_width;
                crop_height = static_cast<size_t>(std::round(crop_width / ASPECT_RATIO_RANGE[0]));
            }
            else if(in_ratio > ASPECT_RATIO_RANGE[1])
            {
                crop_height = original_image_height;
                crop_width  = static_cast<size_t>(std::round(crop_height * ASPECT_RATIO_RANGE[1]));
            }
            else // Whole Image
            {
                crop_height = original_image_height;
                crop_width  = original_image_width;
            }
            double scale = std::min(1.0f, max_area / (crop_width * crop_height));
            crop_width = std::max<int>(1, crop_width * std::sqrt(scale));
            crop_height = std::max<int>(1, crop_height * std::sqrt(scale));
            x1 = std::uniform_int_distribution<int>(0, original_image_width - crop_width)(_rand_gen);
            y1 = std::uniform_int_distribution<int>(0, original_image_height - crop_height)(_rand_gen);
        }
    }
    if(crop_width > max_decoded_width)
        crop_width = max_decoded_width;
    if(crop_height > max_decoded_height)
        crop_height = max_decoded_height;
    //TODO : Turbo Jpeg supports multiple color packing and color formats, add more as an option to the API TJPF_RGB, TJPF_BGR, TJPF_RGBX, TJPF_BGRX, TJPF_RGBA, TJPF_GRAY, TJPF_CMYK , ...
    if( tjDecompress2_partial(m_jpegDecompressor,
                      input_buffer,
                      input_size,
                      output_buffer,
                      max_decoded_width,
                      max_decoded_width * planes,
                      max_decoded_height,
                      tjpf,
                      TJFLAG_FASTDCT, &x1_diff, &crop_width_diff,
 		                  x1, y1, crop_width, crop_height) != 0)
    {
        WRN("Jpeg image decode failed " + STR(tjGetErrorStr2(m_jpegDecompressor)))
        return Status::CONTENT_DECODE_FAILED;
    }

    if (x1 != x1_diff) {
        //std::cout << "x_off changed by tjpeg decoder " << x1 << " " << x1_diff << std::endl;
        unsigned char *src_ptr_temp, *dst_ptr_temp;
        unsigned int elements_in_row = max_decoded_width * planes;
        unsigned int elements_in_crop_row = crop_width * planes;
        //unsigned int remainingElements =  elements_in_row - elements_in_crop_row;
        unsigned int xoffs = (x1-x1_diff) * planes;   // in case x1 gets adjusted by tjpeg decoder
        src_ptr_temp = output_buffer;
        dst_ptr_temp = output_buffer;
        for (unsigned int i = 0; i < crop_height; i++)
        {
            memcpy(dst_ptr_temp, src_ptr_temp + xoffs, elements_in_crop_row * sizeof(unsigned char));
            //memset(dst_ptr_temp + elements_in_crop_row, 0, remainingElements * sizeof(unsigned char));
            src_ptr_temp +=  elements_in_row;
            dst_ptr_temp +=  elements_in_row;
        }
    }
    actual_decoded_width = crop_width;
    actual_decoded_height = crop_height;

    return Status::OK;
}

FusedCropTJDecoder::~FusedCropTJDecoder() {
    tjDestroy(m_jpegDecompressor);
}

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

#pragma once
#include "decoder.h"
#include <turbojpeg.h>
#include <random>

// todo:: move this to common header
template<typename T = std::mt19937, std::size_t state_size = T::state_size>
class SeededRNG {
  /*
  * @param batch_size How many RNGs to store
  * @param state_size How many seed are used to initialize one RNG. Used to lower probablity of
  * collisions between seeds used to initialize RNGs in different operators.
  */
public:
  SeededRNG (int batch_size = 256) {
      std::random_device source;
      _batch_size = batch_size;
      std::size_t _random_data_size = state_size * batch_size ;
      std::vector<std::random_device::result_type> random_data(_random_data_size);
      std::generate(random_data.begin(), random_data.end(), std::ref(source));
      _rngs.reserve(batch_size);
      for (int i=0; i < (int)(_batch_size*state_size); i += state_size) {
        std::seed_seq seeds(std::begin(random_data) + i, std::begin(random_data)+ i +state_size);
        _rngs.emplace_back(T(seeds));
      }
  }

  /**
   * Returns engine corresponding to given sample ID
   */
   T &operator[](int sample) noexcept {
    return _rngs[sample % _batch_size];
  }

private:
    std::vector<T> _rngs;
    int _batch_size;
};

class FusedCropTJDecoder : public Decoder {
public:
    //! Default constructor
    FusedCropTJDecoder();
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
                           Decoder::ColorFormat desired_decoded_color_format, DecoderConfig config, bool keep_original_size=false, uint sample_idx = 0) override;


    ~FusedCropTJDecoder() override;
    void initialize(int device_id) override {};
    bool is_partial_decoder() { return _is_partial_decoder; };
    void set_bbox_coords(std::vector <float> bbox_coord) override { _bbox_coord = bbox_coord;};
    std::vector <float> get_bbox_coords() { return _bbox_coord;}

private:
    tjhandle m_jpegDecompressor;
    const static unsigned SCALING_FACTORS_COUNT =  16;
    const tjscalingfactor SCALING_FACTORS[SCALING_FACTORS_COUNT] = {
            { 2, 1 },
            { 15, 8 },
            { 7, 4 },
            { 13, 8 },
            { 3, 2 },
            { 11, 8 },
            { 5, 4 },
            { 9, 8 },
            { 1, 1 },
            { 7, 8 },
            { 3, 4 },
            { 5, 8 },
            { 1, 2 },
            { 3, 8 },
            { 1, 4 },
            { 1, 8 }
    };
    bool _is_partial_decoder = true;
    std::vector <float> _bbox_coord;
    SeededRNG<std::mt19937, 4> _rngs;     // setting the state_size to 4 for 4 random parameters.
};

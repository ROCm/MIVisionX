#pragma once

#include "decoder.h"
#include <turbojpeg.h>

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
                           Decoder::ColorFormat desired_decoded_color_format, DecoderConfig config, bool keep_original_size=false) override;


    ~FusedCropTJDecoder() override;

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
};